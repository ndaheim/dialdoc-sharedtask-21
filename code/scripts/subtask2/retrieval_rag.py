# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RAG Retriever model implementation."""

import logging
import os
import random
import pickle
import time

import numpy as np
import psutil
import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from transformers import DPRQuestionEncoderTokenizer
from transformers import T5Tokenizer


logger = logging.getLogger(__name__)


class Index(object):
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_doc_dicts(self, doc_ids):
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`):
                A tensor of document indices.
        """
        pass

    def get_top_docs(self, query_vectors, n_docs):
        """
        For each query in the batch, retrieves ``n_docs`` documents.

        Args:
            query_vectors (:obj:`np.array` of shape :obj:`(batch_size, vector_size):
                An array of query vectors.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Returns:
            :obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            :obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns :obj:`True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG model.
        E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load the index.
        """
        raise NotImplementedError


class DocumentSectionIndex(Index):

    def __init__(self, doc_data):
        self.doc_data = doc_data


class StaticKnowledgeIndex(Index):
    def __init__(self, nbest_list):
        self.nbest_list = nbest_list

    def is_initialized(self):
        return True

    def init_index(self):
        pass

    def get_doc_dicts(self, doc_ids):
        doc_dicts = []
        for doc_id_list in doc_ids:
            doc_dict = []
            for doc_id in doc_id_list:
                doc_dict.append(self.knowledge_reader.get_doc_by_id(self.infos[doc_id]['key'][0]))
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_doc_dicts_from_labels(self, batch_doc_labels):
        doc_dicts = []
        for doc_labels in batch_doc_labels:
            doc_dict = []
            for label in doc_labels:
                doc_dict.append(self.knowledge_reader.get_doc(domain=label['domain'], entity_id=label['entity_id'], doc_id=label['doc_id']))
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_top_docs(self, query_vectors, n_docs: int = 5):
        raise Exception('Not implemented')

class KnowledgeIndex(Index):
    def __init__(self, vector_size, embeddings, infos, knowledge_reader):
        self.vector_size = vector_size
        # Dim (batch_size, vector_size)
        self.embeddings = torch.cat([e.cpu() for e in embeddings])
        self.infos = infos
        assert len(self.embeddings) == len(self.infos)

        self.knowledge_reader = knowledge_reader
        self.additional_infos = {}

        self.index_by_key = {
            info['key'][0]: i for i, info in enumerate(infos)
        }

    def is_initialized(self):
        return True

    def init_index(self):
        pass

    def get_doc_dicts(self, doc_ids):
        doc_dicts = []
        for doc_id_list in doc_ids:
            doc_dict = []
            for doc_id in doc_id_list:
                doc_dict.append(self.knowledge_reader.get_doc_by_id(self.infos[doc_id]['key'][0]))
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_top_docs(self, query_vectors, n_docs: int = 5):
        batch_size, embedding_size = query_vectors.shape
        assert embedding_size == self.vector_size
        num_docs = self.embeddings.shape[-2]

        if self.embedding_loss == "triplet":
            dists = torch.cdist(self.embeddings.unsqueeze(0), query_vectors.view(1, batch_size, -1))
            scores = -dists.view(-1, batch_size).transpose(0, 1)
        elif self.embedding_loss == "nll":    
            scores = torch.matmul(
                query_vectors.view(batch_size, 1, embedding_size),
                torch.transpose(self.embeddings, 0, 1).view(1, embedding_size, num_docs)
            ).view(batch_size, -1)
        else:
            raise Exception(f"Unsupported loss: {self.config.embedding_loss}")

        best_indices = scores.topk(n_docs).indices

        if self.additional_infos is not None and 'knowledge_keys' in self.additional_infos:
            for i, knowledge_key in zip(range(len(best_indices)), self.additional_infos['knowledge_keys']):
                correct_knowledge_index = self.index_by_key[knowledge_key]
                if correct_knowledge_index not in best_indices[i]:
                    best_indices[i][-1] = correct_knowledge_index

        vectors = []
        for batch in best_indices:
            batch_vectors = []
            for idx in batch:
                batch_vectors.append(self.embeddings[idx].numpy())
            vectors.append(batch_vectors)

        return best_indices, torch.tensor(vectors)

class RagRetriever(object):
    """
    A distributed retriever built on top of the ``torch.distributed`` communication package. During training all workers
    initalize their own instance of the retriever, however, only the main worker loads the index into memory. The index is stored
    in cpu memory. The index will also work well in a non-distributed setup.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which ``Index`` to build.
    """

    def __init__(self, config, index=None):
        super().__init__()

        if index is None:
            raise Exception("HF and Legacy indices aren't supported.")
        else:
            self.retriever = index

        self.generator_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_generator_tokenizer_name_or_path)
        # TODO(piktus): To be replaced with AutoTokenizer once it supports DPRQuestionEncoderTokenizer
        if config.pretrained_question_encoder_tokenizer_name_or_path is not None:        
            self.question_encoder_tokenizer = AutoTokenizer.from_pretrained(
                config.pretrained_question_encoder_tokenizer_name_or_path
            )
        else:
            self.question_encoder_tokenizer = self.generator_tokenizer
        
        self.process_group = None
        self.n_docs = config.n_docs
        # No chunking
        self.batch_size = 999999  
        self.config = config

        self.is_static = isinstance(self.retriever, StaticKnowledgeIndex)

    def init_retrieval(self, distributed_port):
        """
        Retrirever initalization function, needs to be called from the training process. The function sets some common parameters
        and environment variables. On top of that, (only) the main process in the process group loads the index into memory.

        If this functin doesn't get called, we assume we're operating in a non-distributed environment and the index gets loaded
        at first query.

        Args:
            distributed_port (:obj:`int`):
                The port on which the main communication of the training run is carried out. We set the port for retrieval-related
                communication as ``distributed_port + 1``.
        """

        logger.info("initializing retrieval")

        # initializing a separate process group for retrievel as the default
        # nccl backend doesn't support gather/scatter operations while gloo
        # is too slow to replace nccl for the core gpu communication
        if dist.is_initialized():
            logger.info("dist initialized")
            # needs to be set manually
            os.environ["GLOO_SOCKET_IFNAME"] = self._infer_socket_ifname()
            # avoid clash with the NCCL port
            os.environ["MASTER_PORT"] = str(distributed_port + 1)
            self.process_group = dist.new_group(ranks=None, backend="gloo")

        # initialize retriever only on the main worker
        if not dist.is_initialized() or self._is_main():
            logger.info("dist not initialized / main")
            self.retriever.init_index()

        # all processes wait untill the retriever is initialized by the main process
        if dist.is_initialized():
            torch.distributed.barrier(group=self.process_group)

    def preprocess_query(self, input_string, title, section_id, cheating=True):
        r"""
        Preprocesses the ``input_id`` by first converting it to string using the ``generator_tokenizer`` and
        then tokenizing it using the ``question_encoder_tokenizer``.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Return:
            :obj:`torch.LongTensor`:
                Tokenized input.
            :obj:`str`:
                Decoded input strings.
        """
        all_sections = self.retriever.doc_data[title]
        sample_size = 8 if cheating else len(all_sections)
        
        if len(all_sections) > sample_size:
            section_keys = random.sample(list(all_sections.keys()), 8)
            if not section_id in section_keys and cheating:
                section_keys[-1] = section_id
            sections = [all_sections[key] for key in section_keys]
        else:
            sections = list(all_sections.values())

        questions = [input_string] * len(sections)

        # # handle prefix for T5
        # if isinstance(self.generator_tokenizer, T5Tokenizer):
        #     for i, s in enumerate(input_strings):
        #         if not s.startswith(prefix):
        #             logger.warning("T5 prefix mismatch in {}".format(s))
        #         if len(input_strings[i]) <= len(prefix):
        #             input_strings[i] = ""
        #         else:
        #             input_strings[i] = input_strings[i][len(prefix) :]

        retriever_inputs = self.question_encoder_tokenizer(
            questions,
            sections,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512
        )
        if cheating:
            retriever_inputs["labels"] = sections.index(self.retriever.doc_data[title][section_id])
        retriever_inputs["docs"] = sections #list(sections.values())
        return retriever_inputs

    def postprocess_docs(self, doc_scores, docs, input_strings, add_eos, prefix, title=None, print_docs=False):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_docs)`):
                Retrieval scores of respective docs - passed for logging.
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            add_eos (:obj:`bool`):
                A boolean flag signalling that eos token needs to be added to the contextualized input.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, documents retrieved during the forward pass will be printed out. Intended for debugging purposes.

        Return:
            :obj:`tuple(tuple(torch.FloatTensor)`:
                a tuple consisting od two elements: contextualized ``input_ids`` and a compatible ``attention_mask``.
        """
        rag_input_strings = []

        for i, input_string in enumerate(input_strings):
            for doc in docs[i]:
                if isinstance(doc, dict):
                    rag_input_strings.append(input_string.replace("</s>", "").replace("<s>", "") + " <doc_context> " + doc["text"])
                elif title is not None:
                    rag_input_strings.append(input_string.replace("</s>", "").replace("<s>", "") + " <title>" + "\t" + title + "\t" + "<doc_context> " + doc)
                else:
                    rag_input_strings.append(input_string.replace("</s>", "").replace("<s>", "") + " <doc_context> " + doc)

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=1024,#self.config.max_combined_length,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(doc_scores.device)

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _is_main(self):
        return dist.get_rank(group=self.process_group) == 0

    def _chunk_tensor(self, t, chunk_size):
        n_chunks = t.shape[0] // chunk_size + int(t.shape[0] % chunk_size > 0)
        return list(torch.chunk(t, n_chunks, dim=0))

    def _scattered(self, scatter_list, target_shape, target_type=torch.float32):
        target_tensor = torch.empty(target_shape, dtype=target_type)
        dist.scatter(target_tensor, src=0, scatter_list=scatter_list, group=self.process_group)
        return target_tensor

    def _infer_socket_ifname(self):
        addrs = psutil.net_if_addrs()
        # a hacky way to deal with varying network interface names
        ifname = next((addr for addr in addrs if addr.startswith("e")), None)
        return ifname

    def _main_retrieve(self, query_vectors):
        query_vectors_batched = self._chunk_tensor(query_vectors, 4)#self.batch_size)
        ids_batched = []
        vectors_batched = []
        for query_vectors in query_vectors_batched:
            start_time = time.time()
            ids, vectors = self.retriever.get_top_docs(query_vectors, self.n_docs)
            logger.debug(
                "index search time: {} sec, batch size {}".format(time.time() - start_time, query_vectors.shape)
            )
            ids_batched.append(ids)
            vectors_batched.append(vectors)
        return torch.cat(ids_batched), torch.cat(vectors_batched)

    def retrieve(self, query_vectors, n_docs, labels=None):
        """
        Retrieves documents for specified ``query_vectors``. The main process, which has the access to the index stored in memory, gathers queries
        from all the processes in the main training process group, performs the retrieval and scatters back the results.

        Args:
            query_vectors (:obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Ouput:
            total_scores (:obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`
                The retrieval scores of the retrieved docs per query.
            total_examples (:obj:`List[dict]`):
                The retrieved examples per query.
        """
        doc_ids, doc_vectors = self._main_retrieve(query_vectors)
        return doc_vectors, self.retriever.get_doc_dicts(doc_ids)
