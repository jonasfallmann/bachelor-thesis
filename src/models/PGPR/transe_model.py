from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from pgpr_utils import *
from data_utils import Dataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        self.dataset_name = args.dataset

        # Initialize entity embeddings.
        if self.dataset_name == ML1M:
            self.initialize_entity_embeddings_ml1m(dataset)
        elif self.dataset_name == LFM1M:
            self.initialize_entity_embeddings_lastfm(dataset)
        elif self.dataset_name in CELL:
            self.initialize_entity_embeddings_amazon(dataset)

        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        if self.dataset_name == ML1M:
            self.initialize_relations_embeddings_ml1m(dataset)
        elif self.dataset_name == LFM1M:
            self.initialize_relations_embeddings_lastfm(dataset)
        elif self.dataset_name in CELL:
            self.initialize_relations_embeddings_amazon(dataset)

        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def initialize_entity_embeddings_lastfm(self, dataset):
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            artist=edict(vocab_size=dataset.artist.vocab_size),
            micro_genre=edict(vocab_size=dataset.micro_genre.vocab_size),
            album=edict(vocab_size=dataset.album.vocab_size),
            genre=edict(vocab_size=dataset.genre.vocab_size),
        )
    def initialize_relations_embeddings_lastfm(self,dataset):
        self.relations = edict(
            listened_to=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            created_by=edict(
                et='artist',
                et_distrib=self._make_distrib(dataset.created_by.et_distrib)),
            in_album=edict(
                et='album',
                et_distrib=self._make_distrib(dataset.in_album.et_distrib)),
            has_genre=edict(
                et='genre',
                et_distrib=self._make_distrib(dataset.has_genre.et_distrib)),
            has_micro_genre=edict(
                et='micro_genre',
                et_distrib=self._make_distrib(dataset.has_micro_genre.et_distrib))
        )

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss_lastfm(self, batch_idxs):
        regularizations = []

        user_idxs = batch_idxs[:, 0]
        song_idxs = batch_idxs[:, 1]
        artist_idxs = batch_idxs[:, 2]
        micro_genre_idxs = batch_idxs[:, 3]
        genre_idxs = batch_idxs[:, 4]
        album_idxs = batch_idxs[:, 5]

        # user + listened -> song
        ul_loss, ul_embeds = self.neg_loss('user', 'listened_to', 'product', user_idxs, song_idxs)
        regularizations.extend(ul_embeds)
        loss = ul_loss

        # song + created_by -> artist
        spr_loss, spr_embeds = self.neg_loss('product', 'created_by', 'artist', song_idxs,
                                             artist_idxs)
        if spr_loss is not None:
            regularizations.extend(spr_embeds)
            loss += spr_loss

        # song + has_micro_genre -> micro_genre
        sar1_loss, mpc1_embeds = self.neg_loss('product', 'has_micro_genre', 'micro_genre', song_idxs,
                                               micro_genre_idxs)
        if sar1_loss is not None:
            regularizations.extend(mpc1_embeds)
            loss += sar1_loss

        # song + has_genre -> genre
        sar2_loss, mpc2_embeds = self.neg_loss('product', 'has_genre', 'genre', song_idxs,
                                               genre_idxs)
        if sar2_loss is not None:
            regularizations.extend(mpc2_embeds)
            loss += sar2_loss

        # song + in_album -> album
        sc_loss, sc_embeds = self.neg_loss('product', 'in_album', 'album', song_idxs, album_idxs)
        if sc_loss is not None:
            regularizations.extend(sc_embeds)
            loss += sc_loss


        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        """
        if self.dataset_name == ML1M:
            return self.compute_loss_ml1m(batch_idxs)
        elif self.dataset_name == LFM1M:
            return self.compute_loss_lastfm(batch_idxs)
        elif self.dataset_name in CELL:
            return self.compute_loss_amazon(batch_idxs)
        else:
            print("Dataset {} not recognized during loss computation".format(self.dataset_name))
            exit(-1)

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]

