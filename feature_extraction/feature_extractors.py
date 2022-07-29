from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from utilities.utils import general_weight_init


class FeatureExtractor(nn.Module, ABC):
    """
    Abstract class representing one of the possible FeatureExtractor models. See also FeatureExtractorFactory.
    """

    def __init__(self):
        super().__init__()
        self.cumulative_loss = 0.
        self.name = "FeatureExtractor"

    def init_parameters(self):
        """
        Initial the Feature Extractor parameters
        """
        pass

    def get_and_reset_loss(self) -> float:
        """
        Reset the loss of the feature extractor and returns the computed value
        :return: loss of the feature extractor
        """
        loss = self.cumulative_loss
        self.cumulative_loss = 0.
        return loss

    @abstractmethod
    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs the feature extraction process of the object.
        """
        pass


class Embedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) only given by its embedding.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, only_positive: bool = False):
        """
        Standard Embedding Layer
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param only_positive: whether the embeddings can be only positive
        """
        super().__init__()
        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.only_positive = only_positive
        self.name = "Embedding"

        self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
        print(f'Built Embedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- max_norm: {self.max_norm}\n'
              f'- only_positive: {self.only_positive}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        embeddings = self.embedding_layer(o_idxs)
        if self.only_positive:
            embeddings = torch.absolute(embeddings)
        return embeddings


class EmbeddingW(Embedding):
    """
    FeatureExtractor that places a linear projection after an embedding layer. Used for sharing weights.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, out_dimension: int = None,
                 use_bias: bool = False):
        """
        :param n_objects: see Embedding
        :param embedding_dim: see Embedding
        :param max_norm: see Embedding
        :param out_dimension: Out dimension of the linear layer. If none, set to embedding_dim.
        :param use_bias: whether to use the bias in the linear layer.
        """
        super().__init__(n_objects, embedding_dim, max_norm)
        self.out_dimension = out_dimension
        self.use_bias = use_bias

        if self.out_dimension is None:
            self.out_dimension = embedding_dim

        self.name = 'EmbeddingW'
        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)

        print(f'Built Embeddingw model \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        super().init_parameters()
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_embed = super().forward(o_idxs)
        return self.linear_layer(o_embed)


class AnchorBasedCollaborativeFiltering(FeatureExtractor):
    """
    Anchor-based Collaborative Filtering by Barkan et al. (https://dl.acm.org/doi/10.1145/3459637.3482056) published at CIKM 2021.
    """

    def __init__(self, n_objects: int, embedding_dim: int, anchors: nn.Parameter, delta_exc: float = 0,
                 delta_inc: float = 0, max_norm: float = None):
        super().__init__()
        """
        NB. delta_inc and delta_exc should be passed only when instantiating this FeatureExtractor for Items.

        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param anchors: nn.Parameters with shape (n_anchors,embedding_dim)
        :param delta_exc: factor multiplied to the exclusiveness loss
        :param delta_inc: factor multiplied to the inclusiveness loss
        :param max_norm: max norm of the l2 norm of the embeddings. 
        """

        self.anchors = anchors
        self.n_anchors = anchors.shape[0]
        self.delta_exc = delta_exc
        self.delta_inc = delta_inc

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        self._acc_exc = 0.
        self._acc_inc = 0.
        self.name = "AnchorBasedCollaborativeFiltering"

        print(f'Built AnchorBasedCollaborativeFiltering module \n'
              f'- n_anchors: {self.n_anchors} \n'
              f'- delta_exc: {self.delta_exc} \n'
              f'- delta_inc: {self.delta_inc} \n')

    def init_parameters(self):
        torch.nn.init.normal_(self.anchors, 0, 1)
        torch.nn.init.normal_(self.embedding_ext.embedding_layer.weight, 0, 1)  # Overriding previous init

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [...,embedding_dim]

        o_dots = o_embed @ self.anchors.T  # [...,n_anchors]

        o_coeff = nn.Softmax(dim=-1)(o_dots)  # [...,n_anchors]

        o_vect = o_coeff @ self.anchors  # [...,embedding_dim]

        # Exclusiveness constraint (BCE)
        exc = - (o_coeff * torch.log(o_coeff)).sum()

        # Inclusiveness constraint
        q_k = o_coeff.reshape(-1, self.n_anchors).sum(axis=0).div(o_coeff.sum())  # [n_anchors]
        inc = - (q_k * torch.log(q_k)).sum()

        self._acc_exc += exc
        self._acc_inc += inc

        return o_vect

    def get_and_reset_loss(self) -> float:
        acc_inc, acc_exc = self._acc_inc, self._acc_exc
        self._acc_inc = self._acc_exc = 0
        return - self.delta_inc * acc_inc + self.delta_exc * acc_exc


class PrototypeEmbedding(FeatureExtractor):
    """
    ProtoMF building block. It represents an object (item/user) given the similarity with the prototypes.
    """

    def __init__(self, n_objects: int, embedding_dim: int, n_prototypes: int = None, use_weight_matrix: bool = False,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft', cosine_type: str = 'shifted',
                 max_norm: float = None):
        """
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        :param use_weight_matrix: Whether to use a linear layer after the prototype layer.
        :param sim_proto_weight: factor multiplied to the regularization loss for prototypes
        :param sim_batch_weight: factor multiplied to the regularization loss for batch
        :param reg_proto_type: type of regularization applied batch-prototype similarity matrix on the prototypes. Possible values are ['max','soft','incl']
        :param reg_batch_type: type of regularization applied batch-prototype similarity matrix on the batch. Possible values are ['max','soft']
        :param cosine_type: type of cosine similarity to apply. Possible values ['shifted','standard','shifted_and_div']
        :param max_norm: max norm of the l2 norm of the embeddings.

        """

        super(PrototypeEmbedding, self).__init__()

        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.use_weight_matrix = use_weight_matrix
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight
        self.reg_proto_type = reg_proto_type
        self.reg_batch_type = reg_batch_type
        self.cosine_type = cosine_type

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        if self.n_prototypes is None:
            self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
            self.n_prototypes = self.embedding_dim
        else:
            self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))

        if self.use_weight_matrix:
            self.weight_matrix = nn.Linear(self.n_prototypes, self.embedding_dim, bias=False)

        # Cosine Type
        if self.cosine_type == 'standard':
            self.cosine_sim_func = nn.CosineSimilarity(dim=-1)
        elif self.cosine_type == 'shifted':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y))
        elif self.cosine_type == 'shifted_and_div':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2
        else:
            raise ValueError(f'Cosine type {self.cosine_type} not implemented')

        # Regularization Batch
        if self.reg_batch_type == 'max':
            self.reg_batch_func = lambda x: - x.max(dim=1).values.mean()
        elif self.reg_batch_type == 'soft':
            self.reg_batch_func = lambda x: self._entropy_reg_loss(x, 1)
        else:
            raise ValueError(f'Regularization Type for Batch {self.reg_batch_func} not yet implemented')

        # Regularization Proto
        if self.reg_proto_type == 'max':
            self.reg_proto_func = lambda x: - x.max(dim=0).values.mean()
        elif self.reg_proto_type == 'soft':
            self.reg_proto_func = lambda x: self._entropy_reg_loss(x, 0)
        elif self.reg_proto_type == 'incl':
            self.reg_proto_func = lambda x: self._inclusiveness_constraint(x)
        else:
            raise ValueError(f'Regularization Type for Proto {self.reg_proto_type} not yet implemented')

        self._acc_r_proto = 0
        self._acc_r_batch = 0
        self.name = "PrototypeEmbedding"

        print(f'Built PrototypeEmbedding model \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- use_weight_matrix: {self.use_weight_matrix} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n'
              f'- reg_proto_type: {self.reg_proto_type} \n'
              f'- reg_batch_type: {self.reg_batch_type} \n'
              f'- cosine_type: {self.cosine_type} \n')

    @staticmethod
    def _entropy_reg_loss(sim_mtx, axis: int):
        o_coeff = nn.Softmax(dim=axis)(sim_mtx)
        entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
        return entropy

    @staticmethod
    def _inclusiveness_constraint(sim_mtx):
        '''
        NB. This method is applied only on a square matrix (batch_size,n_prototypes) and it return the negated
        inclusiveness constraints (its minimization brings more equal load sharing among the prototypes)
        '''
        o_coeff = nn.Softmax(dim=1)(sim_mtx)
        q_k = o_coeff.sum(axis=0).div(o_coeff.sum())  # [n_prototypes]
        entropy_q_k = - (q_k * torch.log(q_k)).sum()
        return - entropy_q_k

    def init_parameters(self):
        if self.use_weight_matrix:
            nn.init.xavier_normal_(self.weight_matrix.weight)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [..., embedding_dim]

        # https://github.com/pytorch/pytorch/issues/48306
        sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.prototypes)  # [..., n_prototypes]

        if self.use_weight_matrix:
            w = self.weight_matrix(sim_mtx)  # [...,embedding_dim]
        else:
            w = sim_mtx  # [..., embedding_dim = n_prototypes]

        # Computing additional losses
        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])

        self._acc_r_batch += self.reg_batch_func(batch_proto)
        self._acc_r_proto += self.reg_proto_func(batch_proto)

        return w

    def get_and_reset_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch


class ConcatenateFeatureExtractors(FeatureExtractor):

    def __init__(self, model_1: FeatureExtractor, model_2: FeatureExtractor, invert: bool = False):
        super().__init__()

        """
        Concatenates the latent dimension (considered in position -1) of two Feature Extractors models.
        :param model_1: a FeatureExtractor model
        :param model_2: a FeatureExtractor model
        :param invert: whether to place the latent representation from the second model on top.
        """

        self.model_1 = model_1
        self.model_2 = model_2
        self.invert = invert

        self.name = 'ConcatenateFeatureExtractors'

        print(f'Built ConcatenateFeatureExtractors model \n'
              f'- model_1: {self.model_1.name} \n'
              f'- model_2: {self.model_2.name} \n'
              f'- invert: {self.invert} \n')

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_repr_1 = self.model_1(o_idxs)
        o_repr_2 = self.model_2(o_idxs)

        if self.invert:
            return torch.cat([o_repr_2, o_repr_1], dim=-1)
        else:
            return torch.cat([o_repr_1, o_repr_2], dim=-1)

    def get_and_reset_loss(self) -> float:
        loss_1 = self.model_1.get_and_reset_loss()
        loss_2 = self.model_2.get_and_reset_loss()
        return loss_1 + loss_2

    def init_parameters(self):
        self.model_1.init_parameters()
        self.model_2.init_parameters()
