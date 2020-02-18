"""
We can utilize the syntactic structure of sentence
to extract style words, glue words / phrases.
"""
import torch
import torch.nn as nn
import math
from .embedder import Embedder
from .encoder import Encoder
from .decode_initer import DecIniter
from .decoder import Decoder
from .beam_searcher import BeamSearcher
from .generator import Generator
from .config import *
from common.constants import DEVICE, Q_TYPE2ID_DICT, SPECIAL_TOKEN2ID
from util.tensor_utils import to_thinnest_padded_tensor
from util.file_utils import load


class FQG(nn.Module):
    def __init__(self, config, emb_mats, emb_dicts, dropout=0.1):
        super().__init__()
        self.config = config
        self.config.n_best = 1
        self.dicts = emb_dicts
        self.dicts["idx2tgt"] = dict([[v, k] for k, v in emb_dicts["word"].items()])
        self.PAD = emb_dicts["word"]["<pad>"]

        # input, output embedder
        self.enc_embedder = Embedder(
            config, emb_mats, emb_dicts, dropout)
        if config.share_embedder:
            self.dec_embedder = self.enc_embedder
        else:
            self.dec_embedder = Embedder(
                config, emb_mats, emb_dicts, dropout)
        self.enc_emb_tags = config.emb_tags
        self.dec_emb_tags = ["word"]

        self.src_vocab_limit = config.emb_config["word"]["emb_size"]

        total_emb_size = self.enc_embedder.get_total_emb_dim(self.enc_emb_tags)

        if self.config.use_clue_info:
            self.clue_threshold = 0.5
            clue_embedding_dim = config.emb_config["is_overlap"]["emb_dim"]
            self.clue_embedder = nn.Embedding(
                num_embeddings=3,  # 0: PAD, 1: not overlap, 2: overlap
                embedding_dim=clue_embedding_dim,
                padding_idx=0)

        if self.config.use_style_info:
            style_embedding_dim = config.emb_config["is_overlap"]["emb_dim"]  # NOTICE
            self.style_emb_mat = nn.Parameter(
                torch.randn(
                    self.config.num_question_style,
                    style_embedding_dim)).to(DEVICE)
            nn.init.xavier_normal_(self.style_emb_mat, math.sqrt(3))

        # encoder
        enc_input_size = total_emb_size
        if self.config.use_clue_info:
            enc_input_size += clue_embedding_dim
        self.encoder = Encoder(config, enc_input_size, dropout)

        # decoder
        dec_input_size = config.emb_config["word"]["emb_dim"]
        self.decoder = Decoder(config, dec_input_size, dropout)
        self.decIniter = DecIniter(config)

        # generator
        self.predict_size = min(config.tgt_vocab_limit, len(emb_dicts["word"]))
        if config.use_refine_copy_tgt or config.use_refine_copy_tgt_src:
            self.predict_size = min(
                config.refined_tgt_vocab_limit, len(emb_dicts["word"]))
        self.generator = Generator(
            config.dec_rnn_size // config.maxout_pool_size,
            self.predict_size)

        if self.config.copy_type in ["soft", "soft-oov"]:
            self.related_words_ids_mat = load(config.related_words_ids_mat_file)
            self.related_words_mask = torch.zeros([self.predict_size, self.predict_size]).to(DEVICE)
            for i in range(self.predict_size):
                related_ids = [i for i in self.related_words_ids_mat[i] if i != -1][:self.config.soft_copy_topN]
                self.related_words_mask[i, related_ids] = 1

    def make_init_att(self, context):
        """
        Create init context attention as zero tensor
        """
        batch_size = context.size(1)  # NOTICE: !!!
        h_size = (
            batch_size,
            self.encoder.hidden_size * self.encoder.num_directions)
        result = context.data.new(*h_size).zero_()
        result.requires_grad = False
        return result

    def forward(self, batch):
        src, src_max_len = to_thinnest_padded_tensor(batch["ans_sent_word_ids"])
        batch_size = src.shape[0]
        src_pad_mask = src.data.eq(self.PAD).float()
        src_pad_mask.requires_grad = False
        src = src.transpose(0, 1)

        tgt, tgt_max_len = to_thinnest_padded_tensor(batch["tgt"])
        tgt = tgt.transpose(0, 1)[:-1]  # exclude last <eos> target from inputs

        # input sentence lengths
        maskS = (torch.ones_like(batch["ans_sent_word_ids"]) *
                 self.PAD != batch["ans_sent_word_ids"]).float()
        lengths = maskS.sum(dim=1)

        # mask input content words in generator vocab
        if self.config.use_vocab_mask:
            vocab_mask = torch.ones(batch_size, self.predict_size).to(DEVICE)
            # input word ids to be masked
            wids_to_mask = batch["ans_sent_word_ids"] * torch.LongTensor(batch["ans_sent_is_content"]).to(DEVICE)  # NOTICE!!! memory problem?
            wids_to_mask[wids_to_mask >= self.predict_size] = SPECIAL_TOKEN2ID["<oov>"]
            # set input content words as zero.
            for ii in range(wids_to_mask.shape[0]):
                vocab_mask[ii][wids_to_mask[ii]] = 0
                vocab_mask[ii][[SPECIAL_TOKEN2ID["<pad>"], SPECIAL_TOKEN2ID["<oov>"], SPECIAL_TOKEN2ID["<sos>"], SPECIAL_TOKEN2ID["<eos>"]]] = 1

        # embedding input
        input_emb = self.enc_embedder(
            batch, "ans_sent", self.enc_emb_tags).transpose(1, 2)
        # input_emb shape: batch_size * padded_seq_len * hidden_dim

        # add clue embedding
        if self.config.use_clue_info:
            y_clue = batch["ans_sent_is_clue"]
            clue_ids = ((y_clue.float() + 1) * maskS).long()
            clue_emb = self.clue_embedder(clue_ids)
            input_emb = torch.cat([input_emb, clue_emb], dim=2)

        # encoding
        input_emb = input_emb.transpose(0, 1)
        # input_emb shape: seq_len * batch_size * hidden_dim
        enc_hidden, context = self.encoder(input_emb, lengths)

        # decoding
        init_att = self.make_init_att(context)
        dec_init_input = [enc_hidden[1]]  # [1] is the last backward hiden, NOTICE: it requires must be BRNN
        if self.config.use_style_info:
            y_style = batch["ques_type_id"]
            y_style_one_hot = torch.eye(len(Q_TYPE2ID_DICT))[y_style].to(DEVICE)
            style_emb = torch.matmul(y_style_one_hot, self.style_emb_mat)
            dec_init_input = [enc_hidden[1], style_emb]
        init_dec_hidden = self.decIniter(dec_init_input).unsqueeze(0)

        # !!! as we only feed output word embedding to decoder
        output_emb = self.dec_embedder.embs["word"](tgt)
        (g_out, c_out, c_gate_out,
         dec_hidden, _attn, _attention_vector) = self.decoder(
             output_emb, init_dec_hidden, context, src_pad_mask, init_att)

        batch_size = g_out.size(1)  # !!!
        g_out_t = g_out.view(-1, g_out.size(2))
        g_prob_t = self.generator(g_out_t)
        g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))
        # g_prob_t shape: [max_batch_output_seq_len, batch_size, predict_size]
        # c_out shape: [max_batch_output_seq_len, batch_size, max_batch_input_seq_len]
        # c_gate_out shape: [max_batch_output_seq_len, batch_size, 1]
        if self.config.use_vocab_mask:
            g_prob_t = g_prob_t * vocab_mask.to(DEVICE)
        if self.config.copy_type in ["soft", "soft-oov"]:
            copyGateOutputs = c_gate_out.view(-1, 3)
            g_out_prob = g_prob_t.view(-1, self.predict_size)
            c_out_prob = c_out.view(-1, c_out.size(2))
            # print("c_out_prob shape, ", c_out_prob.shape)  # [max_batch_output_seq_len * batch_size, max_batch_input_seq_len]
            # print("src.transpose(0, 1).shape, ", src.transpose(0, 1).shape)  # [batch_size, max_batch_input_seq_len]

            soft_cp_prob = self.related_words_mask[src.transpose(0, 1).repeat(c_out.size(0), 1)] * \
                c_out_prob.unsqueeze(1).transpose(1, 2).repeat(1, 1, self.predict_size)
            soft_cp_prob = soft_cp_prob.sum(1)
            soft_cp_prob = soft_cp_prob / soft_cp_prob.sum(1).unsqueeze(1).repeat(1, self.predict_size)
            # print("copyGateOutputs.shape: ", copyGateOutputs.shape)  # [max_batch_output_seq_len * batch_size, 3]
            # print("g_out_prob.shape: ", g_out_prob.shape)  # [max_batch_output_seq_len * batch_size, vocab_size]
            # print("soft_cp_prob.shape: ", soft_cp_prob.shape)  # [max_batch_output_seq_len * batch_size, vocab_size]
            new_g_prob_t = g_out_prob * (copyGateOutputs[:, 1].unsqueeze(1).expand_as(g_out_prob)) + \
                soft_cp_prob * (copyGateOutputs[:, 2].unsqueeze(1).expand_as(g_out_prob))

            soft_cp_prob = soft_cp_prob.view(g_prob_t.shape)  # prob of soft-copied words
            new_g_prob_t = new_g_prob_t.view(g_prob_t.shape)  # prob of both generated and soft-copied words
            # print("soft_cp_prob.shape,\n ", soft_cp_prob.shape)  # [max_batch_output_seq_len, batch_size, vocab_size]
            # print("new_g_prob_t.shape,\n ", new_g_prob_t.shape)  # [max_batch_output_seq_len, batch_size, vocab_size]

        if self.config.copy_type in ["soft", "soft-oov"]:
            result = (g_prob_t, c_out, c_gate_out, src_max_len, soft_cp_prob, new_g_prob_t)
        else:
            result = (g_prob_t, c_out, c_gate_out, src_max_len)

        return result

    def buildTargetTokens(self, pred, src, isCopy, copyPosition, attn):
        pred_word_ids = [x.item() for x in pred]
        tokens = []

        # get generated tokens
        for i in pred_word_ids:
            tokens.append(self.dicts["idx2tgt"].get(i))
            if i == self.dicts["word"]["<eos>"]:
                break
        tokens = tokens[:-1]  # delete EOS

        # get copied input tokens
        for i in range(len(tokens)):
            if isCopy[i]:
                tokens[i] = '[[{0}]]'.format(
                    src[copyPosition[i] - self.predict_size])  # NOTICE: Beam search append input words after vocab words

        # replace unknown words with the input token that has maximum attn
        for i in range(len(tokens)):
            if tokens[i] == "<oov>":
                _, maxIndex = attn[i].max(0)
                # print("maxIndex is: ", maxIndex)  # NOTICE: always 0 ???
                tokens[i] = src[maxIndex.item()].encode("utf8").decode("utf8")  # NOTICE: added item() here..
        return tokens

    def translate_batch(self, batch):
        src, src_max_len = to_thinnest_padded_tensor(
            batch["ans_sent_word_ids"])
        src = src.transpose(0, 1)
        batch_size = src.size(1)
        beam_size = self.config.beam_size

        #  (1) run the encoder on the src
        # input sentence lengths
        maskS = (torch.ones_like(batch["ans_sent_word_ids"]) *
                 self.PAD != batch["ans_sent_word_ids"]).float()
        lengths = maskS.sum(dim=1)

        # mask input content words in generator vocab
        if self.config.use_vocab_mask:
            vocab_mask = torch.ones(batch_size, self.predict_size).to(DEVICE)
            # input word ids to be masked
            wids_to_mask = batch["ans_sent_word_ids"] * torch.LongTensor(batch["ans_sent_is_content"]).to(DEVICE)  # NOTICE!!! memory problem?
            wids_to_mask[wids_to_mask >= self.predict_size] = SPECIAL_TOKEN2ID["<oov>"]
            # set input content words as zero.
            for ii in range(wids_to_mask.shape[0]):
                vocab_mask[ii][wids_to_mask[ii]] = 0
                vocab_mask[ii][[SPECIAL_TOKEN2ID["<pad>"], SPECIAL_TOKEN2ID["<oov>"], SPECIAL_TOKEN2ID["<sos>"], SPECIAL_TOKEN2ID["<eos>"]]] = 1

        input_emb = self.enc_embedder(
            batch, "ans_sent", self.enc_emb_tags).transpose(1, 2)

        # add clue embedding
        if self.config.use_clue_info:
            y_clue = batch["ans_sent_is_clue"]
            clue_ids = ((y_clue.float() + 1) * maskS).long()
            clue_emb = self.clue_embedder(clue_ids)
            input_emb = torch.cat([input_emb, clue_emb], dim=2)

        # (2) encoding
        input_emb = input_emb.transpose(0, 1)
        enc_hidden, context = self.encoder(input_emb, lengths)
        # enc_hidden - hidden representation for the whole input. Two layers for BiGRU.
        #     Shape [num_layers, batch_size, hidden_dim]
        # context - context representation for each input token
        #     Shape [unpadded_seq_len, batch_size, hidden_dim * num_layers]

        # (3) run the decoder to generate sentences, using beam search
        dec_init_input = [enc_hidden[1]]
        y_style = None
        if self.config.use_style_info:
            y_style = batch["ques_type_id"]
            y_style_one_hot = torch.eye(len(Q_TYPE2ID_DICT))[y_style].to(DEVICE)
            style_emb = torch.matmul(y_style_one_hot, self.style_emb_mat)
            dec_init_input = [enc_hidden[1], style_emb]
        dec_hidden = self.decIniter(dec_init_input)  # [batch_size, dec_rnn_size]
        # Expand tensors for each beam.
        context = context.data.repeat(1, beam_size, 1)  # NOTICE
        dec_hidden = dec_hidden.unsqueeze(0).data.repeat(1, beam_size, 1)  # NOTICE   any bug here ?????
        att_vec = self.make_init_att(context)
        padMask = src.data.eq(self.dicts["word"]["<pad>"]).transpose(
            0, 1).unsqueeze(0).repeat(beam_size, 1, 1).float()

        beam = [BeamSearcher(beam_size) for k in range(batch_size)]  # each sample has a beam searcher
        batchIdx = list(range(batch_size))
        remainingSents = batch_size  # count how many remaining samples need to generate output
        remainingSents_batch_ids = list(range(batch_size))

        # decode over each time step
        for i in range(self.config.sent_limit):  # NOTICE: !!! maximum output length: sent_limit
            # Prepare decoder input.
            input = torch.stack(  # [1, remainingSents * beam_size]
                [b.getCurrentState() for b in beam
                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)

            output_emb = self.dec_embedder.embs["word"](input)  # [1, remainingSents * beam_size, word_emb_dim]

            g_outputs, c_outputs, copyGateOutputs, dec_hidden, attn, att_vec = \
                self.decoder(
                    output_emb, dec_hidden, context,
                    padMask.view(-1, padMask.size(2)), att_vec)  # NOTICE: !!! debug mode, the word emb don't have 20000 unique words, therefore, it will cause index out-of-range error.

            # g_outputs: 1 x (beam*batch) x numWords
            if self.config.copy_type in ["soft", "soft-oov"]:
                # print("copyGateOutputs shape: ", copyGateOutputs.shape)  # [1, remain_batch_size * beam_size, 3]
                # print("attn shape: ", attn.shape)  # [remain_batch_size * beam_size, max_batch_input_size]
                copyGateOutputs = copyGateOutputs.view(-1, 3)
                g_outputs = g_outputs.squeeze(0)
                g_out_prob = self.generator.forward(g_outputs) + 1e-8
                # print("c_outputs.shape: ",c_outputs.shape)  # [1, remain_batch_size * beam_size, max_batch_input_size]
                if self.config.use_vocab_mask:
                    g_out_prob = g_out_prob * vocab_mask[remainingSents_batch_ids].repeat((beam_size, 1)).to(DEVICE) + 1e-8  # NOTICE: make sure mask matches!!!
                soft_cp_prob = self.related_words_mask[src.transpose(0, 1)[remainingSents_batch_ids].repeat(beam_size, 1)] * \
                    c_outputs.squeeze(0).unsqueeze(1).transpose(1, 2).repeat(1, 1, self.predict_size)
                soft_cp_prob = soft_cp_prob.sum(1)
                soft_cp_prob = soft_cp_prob / soft_cp_prob.sum(1).unsqueeze(1).repeat(1, self.predict_size)
                # print("g_out_prob sum 1: \n", g_out_prob.sum(1))
                # print("soft_cp_prob sum 1: \n", soft_cp_prob.sum(1))

                g_predict = torch.log(
                    g_out_prob * (copyGateOutputs[:, 1].unsqueeze(1).expand_as(g_out_prob)) +
                    soft_cp_prob * (copyGateOutputs[:, 2].unsqueeze(1).expand_as(g_out_prob)))  # NOTICE: log

                c_outputs = c_outputs.squeeze(0) + 1e-8
                # print("c_outputs sum 1: \n", c_outputs.sum(1))
                c_predict = torch.log(c_outputs * (copyGateOutputs[:, 0].unsqueeze(1).expand_as(c_outputs)))  # NOTICE: log
            else:
                copyGateOutputs = copyGateOutputs.view(-1, 1)
                g_outputs = g_outputs.squeeze(0)
                g_out_prob = self.generator.forward(g_outputs) + 1e-8
                if self.config.use_vocab_mask:
                    g_out_prob = g_out_prob * vocab_mask[remainingSents_batch_ids].repeat((beam_size, 1)) + 1e-8  # NOTICE: make sure mask matches!!!
                g_predict = torch.log(g_out_prob * ((1 - copyGateOutputs).expand_as(g_out_prob)))  # NOTICE: log
                c_outputs = c_outputs.squeeze(0) + 1e-8
                c_predict = torch.log(c_outputs * (copyGateOutputs.expand_as(c_outputs)))  # NOTICE: log
                # print("g_predict: ", g_predict.shape)  # [remainingSents * beam_size, vocabulary_size]
                # print("c_predict: ", c_predict.shape)  # [remainingSents * beam_size, src_max_len]
                # print("copyGateOutputs: ", copyGateOutputs.shape)  # [remainingSents * beam_size, 1]

            # batch x beam x numWords
            vocab_words_loglike = g_predict.view(beam_size, remainingSents, -1).transpose(0, 1).contiguous()
            copy_words_loglike = c_predict.view(beam_size, remainingSents, -1).transpose(0, 1).contiguous()
            # we can see that copy_words_loglike(i) = log( attn(i) * copyGateOutputs(i) ) for i-th input token
            attn = attn.view(beam_size, remainingSents, -1).transpose(0, 1).contiguous()
            # print("vocab_words_loglike: ", vocab_words_loglike.shape)  # [remainingSents, beam_size, vocabulary_size]
            # print("copy_words_loglike: ", copy_words_loglike.shape)  # [remainingSents, beam_size, src_max_len]
            # print("attn: ", attn.shape)  # [remainingSents, beam_size, src_max_len]

            active = []
            father_idx = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batchIdx[b]  # batchIdx is map <beam_idx (1~batch_size), current idx in remained sents>
                if not beam[b].advance(vocab_words_loglike.data[idx], copy_words_loglike.data[idx], attn.data[idx]):  # if not done (done is get <eos>)
                    active += [b]  # record the still active batch index
                    father_idx.append(beam[b].father_beam_idxs[-1])  # NOTICE: this is very annoying

            if not active:  # if active is empty []
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = torch.LongTensor(
                [batchIdx[k] for k in active]).to(DEVICE)
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            dec_hidden = updateActive(dec_hidden, self.config.dec_rnn_size)
            context = updateActive(context, self.config.enc_rnn_size)
            att_vec = updateActive(att_vec, self.config.enc_rnn_size)
            padMask = padMask.index_select(1, activeIdx)

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(
                0, 1).contiguous()
            dec_hidden = dec_hidden.view(-1, dec_hidden.size(2)).index_select(
                0, previous_index.view(-1)).view(*dec_hidden.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(
                0, previous_index.view(-1)).view(*att_vec.size())

            remainingSents = len(active)
            remainingSents_batch_ids = active

        # (4) package everything up
        allHyp, allScores, allAttn, allIsCopy, allCopyPosition = [], [], [], [], []
        n_best = self.config.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sortBest()
            allScores += [scores[:n_best]]
            valid_attn = src.data[:, b].ne(self.dicts["word"]["<pad>"]).nonzero().squeeze(1)
            hyps, isCopy, copyPosition, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]
            allIsCopy += [isCopy]
            allCopyPosition += [copyPosition]

        #  (3) convert indexes to words
        predBatch = []
        src_batch = batch["src_tokens"]
        for b in range(batch_size):
            n = 0
            predBatch.append(
                self.buildTargetTokens(
                    allHyp[b][n], src_batch[b], allIsCopy[b][n],
                    allCopyPosition[b][n], allAttn[b][n])
            )
        return predBatch, maskS
