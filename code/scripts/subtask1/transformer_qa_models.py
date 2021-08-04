import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig, RobertaModel, RobertaConfig, \
  AlbertPreTrainedModel, AlbertModel
from torch.nn import CrossEntropyLoss, MSELoss
import gc
import itertools

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class Biaffine(nn.Module):
  r"""
  Biaffine layer for first-order scoring.
  This function has a tensor of weights :math:`W` and bias terms if needed.
  The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
  in which :math:`x` and :math:`y` can be concatenated with bias terms.
  References:
      - Timothy Dozat and Christopher D. Manning. 2017.
        `Deep Biaffine Attention for Neural Dependency Parsing`_.
  Args:
      n_in (int):
          The size of the input feature.
      n_out (int):
          The number of output channels.
      bias_x (bool):
          If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
      bias_y (bool):
          If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
  .. _Deep Biaffine Attention for Neural Dependency Parsing:
      https://openreview.net/forum?id=Hk95PK9le
  """

  def __init__(self, n_in, n_hidden, n_out=1, bias_x=True, bias_y=True):
    super().__init__()

    self.n_in = n_in
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.bias_x = bias_x
    self.bias_y = bias_y
    self.weight = nn.Parameter(torch.randn(n_out, n_hidden + bias_x, n_hidden + bias_y))

    self.start_proj = nn.Linear(n_in, n_hidden)
    self.end_proj = nn.Linear(n_in, n_hidden)

    #self.reset_parameters()

  def __repr__(self):
    s = f"n_in={self.n_in}, n_out={self.n_out}"
    if self.bias_x:
      s += f", bias_x={self.bias_x}"
    if self.bias_y:
      s += f", bias_y={self.bias_y}"

    return f"{self.__class__.__name__}({s})"

  def reset_parameters(self):
    nn.init.zeros_(self.weight)

  def forward(self, states):
    r"""
    Args:
        states (torch.Tensor): ``[batch_size, seq_len, n_in]``.
    Returns:
        ~torch.Tensor:
            A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
            If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
    """
    x = self.start_proj(states)
    y = self.end_proj(states)
    if self.bias_x:
      x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
    if self.bias_y:
      y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
    # [batch_size, n_out, seq_len, seq_len]
    s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
    # remove dim 1 if n_out == 1
    s = s.squeeze(1)

    return s


class BiAffine(nn.Module):
  def __init__(self, hidden_size, label_size):
    super().__init__()

    self.parse_proj = nn.Parameter(
      torch.randn(1, label_size, hidden_size, hidden_size))
    self.offset_proj = nn.Parameter(
      torch.randn(hidden_size, label_size))
    self.offset = nn.Parameter(torch.randn(label_size))

  def forward(self, sent_states):
    label_size = self.parse_proj.size(1)
    batch_size = sent_states.size(0)
    max_len = sent_states.size(1)
    hidden_size = sent_states.size(2)
    sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
    sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
    parse_proj = self.parse_proj

    # project to CRF potentials

    # binear part
    # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
    proj = torch.matmul(sent_states, parse_proj)
    # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
    log_potentials = torch.matmul(proj, sent_states_)
    # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
    log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
    # [[batch, len * len, label] -> [[batch, len, len, label]
    log_potentials_0 = log_potentials.view(batch_size, max_len, max_len, label_size)

    # local offset
    sent_states_sum_0 = sent_states.view(batch_size, max_len, 1, hidden_size)
    sent_states_sum_1 = sent_states.view(batch_size, 1, max_len, hidden_size)
    # [batch, len, 1, hidden] + [batch, 1, len, hidden] -> [batch, len, len, hidden]
    sent_states_sum = (sent_states_sum_0 + sent_states_sum_1).view(batch_size, -1, hidden_size)
    offset_proj = self.offset_proj.view([1, hidden_size, -1])
    # [batch, len * len, hidden] * [1, hidden, label] -> [batch, len * len, label]
    log_potentials_1 = torch.matmul(sent_states_sum, offset_proj)
    log_potentials_1 = log_potentials_1.view(batch_size, max_len, max_len, label_size)

    offset = self.offset.view(1, 1, 1, label_size)
    log_potentials = log_potentials_0 + log_potentials_1 + offset
    return log_potentials


class ElectraForSpanQuestionAnswering(ElectraPreTrainedModel):
  config_class = ElectraConfig
  base_model_prefix = "electra"

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.electra = ElectraModel(config)
    self.span_output = Biaffine(config.hidden_size, config.biaffine_hidden_size, 1)

    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      start_positions=None,
      end_positions=None,
      span_starts=None,
      span_ends=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""
    start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    """
    return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict

    discriminator_hidden_states = self.electra(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
    )

    sequence_output = discriminator_hidden_states[0]

    log_potentials = self.span_output(sequence_output)
    log_potentials = log_potentials.squeeze(-1)

    # Mask invalid potentials
    batch_size = log_potentials.size(0)
    max_seq_length = log_potentials.size(1)

    denominator_mask = torch.triu(torch.ones_like(log_potentials))
    denominator_mask = torch.logical_and(
      attention_mask.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length)),
      denominator_mask)

    if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
      start_mask = span_starts.view(batch_size, max_seq_length, 1).expand((batch_size, max_seq_length, max_seq_length))
      end_mask = span_ends.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length))
      span_mask = torch.logical_and(start_mask, end_mask)

      denominator_mask = torch.logical_and(span_mask, denominator_mask)

    log_potentials[denominator_mask < 1] = -float('inf')

    total_loss = None
    if start_positions is not None and end_positions is not None:
      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      loss_mask = torch.logical_or(start_positions >= max_seq_length, end_positions >= max_seq_length)
      start_positions = torch.clamp(start_positions, 0, max_seq_length - 1)
      end_positions = torch.clamp(end_positions, 0, max_seq_length - 1)

      # (batch_size,)
      idx = torch.range(0, batch_size - 1, device=start_positions.device, dtype=start_positions.dtype) * (max_seq_length * max_seq_length) + \
          start_positions * max_seq_length + end_positions

      log_scores = log_potentials.view(-1)[idx]
      nominator = torch.logsumexp(log_potentials, (1, 2))
      loss = log_scores - nominator
      loss = torch.where(loss_mask, torch.zeros_like(loss), loss)
      total_loss = -torch.mean(loss)

    output = (log_potentials,) + discriminator_hidden_states[1:]
    return ((total_loss,) + output) if total_loss is not None else output


class RobertaForSpanQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.span_output = Biaffine(config.hidden_size, config.biaffine_hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_starts=None,
        span_ends=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        log_potentials = self.span_output(sequence_output)
        log_potentials = log_potentials.squeeze(-1)

        # Mask invalid potentials
        batch_size = log_potentials.size(0)
        max_seq_length = log_potentials.size(1)

        denominator_mask = torch.triu(torch.ones_like(log_potentials))
        denominator_mask = torch.logical_and(
          attention_mask.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length)),
          denominator_mask)

        if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
          start_mask = span_starts.view(batch_size, max_seq_length, 1).expand((batch_size, max_seq_length, max_seq_length))
          end_mask = span_ends.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length))
          span_mask = torch.logical_and(start_mask, end_mask)
          denominator_mask = torch.logical_and(span_mask, denominator_mask)

        log_potentials[denominator_mask < 1] = -float('inf')

        total_loss = None
        if start_positions is not None and end_positions is not None:
          # If we are on multi-GPU, split add a dimension
          if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
          if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
          # sometimes the start/end positions are outside our model inputs, we ignore these terms
          loss_mask = torch.logical_or(start_positions >= max_seq_length, end_positions >= max_seq_length)
          start_positions = torch.clamp(start_positions, 0, max_seq_length - 1)
          end_positions = torch.clamp(end_positions, 0, max_seq_length - 1)

          idx = torch.range(0, batch_size - 1, device=start_positions.device, dtype=start_positions.dtype) * (max_seq_length * max_seq_length) + \
              start_positions * max_seq_length + end_positions

          # TODO handle ignored_index
          log_scores = log_potentials.view(-1)[idx]
          # TODO only use relevant scores in the denominator
          nominator = torch.logsumexp(log_potentials, (1, 2))
          loss = log_scores - nominator
          loss = torch.where(loss_mask, torch.zeros_like(loss), loss)
          total_loss = -torch.mean(loss)

        output = (log_potentials,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output



class ElectraForDefaultQuestionAnswering(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_starts=None,
        span_ends=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
          start_logits[span_starts < 1] = -float('inf')
          end_logits[span_ends < 1] = -float('inf')

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (
            logits,
        ) + discriminator_hidden_states[1:]
        return ((total_loss,) + output) if total_loss is not None else output



class RobertaForDefaultQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_starts=None,
        span_ends=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
          start_logits[span_starts < 1] = -float('inf')
          end_logits[span_ends < 1] = -float('inf')

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (logits,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class AlbertForDefaultQuestionAnswering(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_starts=None,
        span_ends=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = True

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
          start_logits[span_starts < 1] = -float('inf')
          end_logits[span_ends < 1] = -float('inf')

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            total_loss = torch.where(torch.isinf(total_loss), torch.zeros_like(total_loss), total_loss)

        output = (logits,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class AlbertForSpanQuestionAnswering(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.span_output = Biaffine(config.hidden_size, config.biaffine_hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_starts=None,
        span_ends=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = True

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        log_potentials = self.span_output(sequence_output)
        log_potentials = log_potentials.squeeze(-1)

        # Mask invalid potentials
        batch_size = log_potentials.size(0)
        max_seq_length = log_potentials.size(1)

        denominator_mask = torch.triu(torch.ones_like(log_potentials))
        denominator_mask = torch.logical_and(
          attention_mask.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length)),
          denominator_mask)


        if self.config.span_restricted and (self.training or not self.config.span_restricted_only_in_eval):
          start_mask = span_starts.view(batch_size, max_seq_length, 1).expand((batch_size, max_seq_length, max_seq_length))
          end_mask = span_ends.view(batch_size, 1, max_seq_length).expand((batch_size, max_seq_length, max_seq_length))
          span_mask = torch.logical_and(start_mask, end_mask)
          denominator_mask = torch.logical_and(span_mask, denominator_mask)

        log_potentials[denominator_mask < 1] = -float('inf')

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            loss_mask = torch.logical_or(start_positions >= max_seq_length, end_positions >= max_seq_length)
            start_positions = torch.clamp(start_positions, 0, max_seq_length - 1)
            end_positions = torch.clamp(end_positions, 0, max_seq_length - 1)

            # (batch_size,)
            idx = torch.range(0, batch_size - 1, device=start_positions.device, dtype=start_positions.dtype) * (max_seq_length * max_seq_length) + \
                start_positions * max_seq_length + end_positions

            log_scores = log_potentials.view(-1)[idx]
            nominator = torch.logsumexp(log_potentials, (1, 2))
            loss = log_scores - nominator
            loss = torch.where(loss_mask, torch.zeros_like(loss), loss)
            total_loss = -torch.mean(loss)

        output = (log_potentials,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
