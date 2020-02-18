import torch
from .config import *
from common.constants import DEVICE, SPECIAL_TOKEN2ID


class BeamSearcher(object):

    def __init__(self, size):

        self.size = size  # beam size
        self.done = False  # whether this beam finished (get <eos>)

        # The score for each translation on the beam.
        self.cur_output_loglike = torch.FloatTensor(size).zero_().to(DEVICE)
        self.output_accumulate_loglikes = []
        self.output_lens = []

        # The backpointers at each time-step.
        self.father_beam_idxs = []

        # The outputs at each time-step.
        self.output_word_ids = [torch.LongTensor(size).fill_(SPECIAL_TOKEN2ID["<pad>"]).to(DEVICE)]  # [tensor([0, 0, 0, 0, 0, ..])]
        self.output_word_ids[0][0] = SPECIAL_TOKEN2ID["<sos>"]  # [tensor([2, 0, 0, 0, 0, ..])]
        self.output_word_ids_true = [torch.LongTensor(size).fill_(SPECIAL_TOKEN2ID["<pad>"]).to(DEVICE)]
        self.output_word_ids_true[0][0] = SPECIAL_TOKEN2ID["<sos>"]
        # NOTICE: why only one beam use <sos> rather than each beam stats with <sos>?
        # That is because in the second timestep, it will natually get top beam_size outputs, each will
        # start with <sos>. So that in the first timestep, we don't need each beam start with <sos>, just one is enough.

        # The attentions (matrix) for each time.
        self.attn = []

        # is copy for each time
        self.isCopy = []

    def getCurrentState(self):
        # Get the outputs for the current time step.
        return self.output_word_ids[-1]

    def getCurrentOrigin(self):
        # Get the back-pointers for the current time step.
        return self.father_beam_idxs[-1]

    def advance(self, vocab_words_loglike, copy_words_loglike, attnOut):
        """
        Given prob over words for every last step beam and attention,
        compute and update the beam search for one step.
        :param vocab_words_loglike: the log likelihood of output vocabulary words from the last step.
        :param copy_words_loglike: the log likelihood of output copied words from the last step.
        :param attnOut: attention at the last step.
        :return: True if beam search is complete.
        """
        vocab_size = vocab_words_loglike.size(1)  # vocab_words_loglike: [beam_size, vocab_size]
        src_max_len = copy_words_loglike.size(1)  # copy_words_loglike: [beam_size, src_max_len]
        vocab_plus_input_size = vocab_size + src_max_len
        vocab_and_input_words_loglike = torch.cat((vocab_words_loglike, copy_words_loglike), dim=1)  # [beam_size, vocab_size + src_max_len] NOTICE: !!this is using copy_oov, so just concat.
        # vocab_and_input_words_loglike are log prob values of each word in vocab and input sequence.

        if len(self.father_beam_idxs) > 0:
            is_beam_finished = self.output_word_ids[-1].eq(SPECIAL_TOKEN2ID["<eos>"])  # 1d binary tensor, indicate which beam is finished.
            if any(is_beam_finished):
                vocab_and_input_words_loglike.masked_fill_(
                    is_beam_finished.unsqueeze(1).expand_as(vocab_and_input_words_loglike),
                    -float('inf'))  # As the score is log. here inf means the prob of masked position is 0.
                for i in range(self.size):
                    if self.output_word_ids[-1][i] == SPECIAL_TOKEN2ID["<eos>"]:
                        vocab_and_input_words_loglike[i][SPECIAL_TOKEN2ID["<eos>"]] = 0  # As the score is log. here 0 means the prob of <eos> is 1.
            # so after above if, vocab_and_input_words_loglike[i] will be [-inf, -inf, -inf, 0, -inf, ...] for beam i.
            # vocab_and_input_words_loglike shape is: beam_size * (vocab_size + src_max_len)

            # set up the current step length
            cur_output_lens = self.output_lens[-1]  # 1d tensor, record the current output length of each beam
            for i in range(self.size):
                cur_output_lens[i] += 0 if self.output_word_ids[-1][i] == SPECIAL_TOKEN2ID["<eos>"] else 1

        # Sum the previous scores.
        if len(self.father_beam_idxs) > 0:
            prev_output_accumulate_loglike = self.output_accumulate_loglikes[-1]
            cur_output_accumulate_loglike = vocab_and_input_words_loglike + prev_output_accumulate_loglike.unsqueeze(1).expand_as(vocab_and_input_words_loglike)
            # current accumulated scores. here we use +, because scores are log probability.
            cur_output_accumulate_scores = cur_output_accumulate_loglike / cur_output_lens.unsqueeze(1).expand_as(cur_output_accumulate_loglike)
            # cur_output_accumulate_scores is cur_output_accumulate_loglike normalized by each beam's output sequence length.
        else:
            self.output_lens.append(torch.FloatTensor(self.size).fill_(1).to(DEVICE))  # init length is [1, 1, 1, 1, 1]
            cur_output_accumulate_scores = vocab_and_input_words_loglike[0]  # init cur_output_accumulate_scores, vocab_and_input_words_loglike[0] [beam_size, vocab_size + max_src_len]

        flat_cur_output_accumulate_scores = cur_output_accumulate_scores.view(-1)  # so the size is beam_size * (vocab_size + max_src_len)

        best_output_accumulate_scores, best_output_accumulate_scores_id = flat_cur_output_accumulate_scores.topk(self.size, 0, True, True)
        # best_output_accumulate_scores - 1d tensor, the top beam_size scores in flat_cur_output_accumulate_scores. descent order.
        # best_output_accumulate_scores_id - 1d tensor, the index of best scores in flat_cur_output_accumulate_scores
        self.cur_output_loglike = best_output_accumulate_scores
        # so self.cur_output_loglike record the current s

        # best_output_accumulate_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        father_beam_idx = best_output_accumulate_scores_id / vocab_plus_input_size
        # father_beam_idx: 1d tensor, beam_size length. It indicates the top beam_size current best scores
        # come from which previous beam. Therefore, each item in it will
        # be <= beam_size -1. For example: tensor([2, 0, 1, 2, 3])
        predict = best_output_accumulate_scores_id - father_beam_idx * vocab_plus_input_size
        # predict - 1d tensor, length=beam_size. It indicates the word index (0 ~ vocab_size + max_src_len)
        # of top beam_size predictions.
        isCopy = predict.ge(torch.LongTensor(self.size).fill_(vocab_size).to(DEVICE)).long()  # NOTICE
        # isCopy - 1d binary tensor, length=beam_size
        # it indicates whether each current top beam_size predict tokens are copied tokens.
        # as we concatenate input tokens with the whole vocabulary, therefore,
        # if predict >= vocab_size(vocab_size = vocab_size + max_src_len), then this token is copied.
        final_predict = predict * (1 - isCopy) + isCopy * SPECIAL_TOKEN2ID["<oov>"]  # NOTICE: !!!!!!! This is used for only hard-oov copy.
        # final_predict - 1d tensor, length=beam_size
        # here we replace copied word index with <oov> index.

        if len(self.father_beam_idxs) > 0:
            self.output_lens.append(cur_output_lens.index_select(0, father_beam_idx))
            # update cur_output_lens.
            # For example, if we know cur_output_lens = [1, 2, 3, 4, 5] (beam_size=5), and now father_beam_idx is [2, 2, 3, 3, 1]
            # then cur_output_lens is [3, 3, 4, 4, 2]. Because previously it is the 5 beam length, now we only
            # keep the beams in the father_beam_idx.
            self.output_accumulate_loglikes.append(cur_output_accumulate_loglike.view(-1).index_select(0, best_output_accumulate_scores_id))
            # similarly, the accumulated scores also only keep the top beam_size scores
        else:
            self.output_accumulate_loglikes.append(self.cur_output_loglike)
            # if this is the first decoder step,
            # then the self.cur_output_loglike are the accumulated scores.
            # So, self.output_accumulate_loglikes records the accumulated best log prob for the whole output sequence
            # on each timestep. Each item in this list is 1d tensor with length = beam_size

        self.father_beam_idxs.append(father_beam_idx)
        self.output_word_ids.append(final_predict)  # output_word_ids stores the predicted word ids. copied words id replaced by oov id
        self.output_word_ids_true.append(predict)  # nexYs_true stores the predicted word ids. copied words id not replaced by oov id
        self.isCopy.append(isCopy)
        self.attn.append(attnOut.index_select(0, father_beam_idx))  # store the top beam_size output's attn weight on each step

        # End condition is when every one is EOS.
        if all(self.output_word_ids[-1].eq(SPECIAL_TOKEN2ID["<eos>"])):
            self.done = True
        return self.done

    def sortBest(self):
        return torch.sort(self.cur_output_loglike, 0, True)
        # actually, cur_output_loglike is already ordered in advance() by topk().

    def getHyp(self, k):
        """
        Get the k-th best output sequence word ids, attn scores, is_copy, copy position.
        """
        hyp, attn, isCopy, copyPos = [], [], [], []
        for j in range(len(self.father_beam_idxs) - 1, -1, -1):
            # Here we start from the last step. The reason is that:
            # in the last step, the k-th output in variables are belonging to the k-th best output.
            # in previous steps, the k-th best output's index is self.father_beam_idxs[j][k].
            hyp.append(self.output_word_ids[j + 1][k])
            attn.append(self.attn[j][k])
            isCopy.append(self.isCopy[j][k])
            copyPos.append(self.output_word_ids_true[j + 1][k])
            k = self.father_beam_idxs[j][k]
        return hyp[::-1], isCopy[::-1], copyPos[::-1], torch.stack(attn[::-1])
        # note that [::-1] reverse the list. Therefore the output is in the correct order.
        # hyp[::-1] - 1d list, each item is a single value tensor of word index
        # isCopy[::-1] - 1d list, each item is a binary value tensor that indicates whether this word is copied from input
        # copyPos[::-1] - 1d list, each item indicates the copy position of copied word
        # torch.stack(attn[::-1]) - 2d tensor, shape [output_len, src_max_len]
