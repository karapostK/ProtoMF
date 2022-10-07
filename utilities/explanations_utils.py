import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(objects: np.ndarray, prototypes: np.ndarray, object_legend_text: str = 'Object', perplexity: int = 5,
              path_save_fig: str = None):
    """
    Creates a TSNE plot to visualize the object embeddings and the prototypes in the same space.
    :param objects: Object (users/items) embedding to plot in the same space of the prototypes
    :param prototypes: Prototypes (users/items) embeddings
    :param object_legend_text: Text to show in the legend for the Object
    :param perplexity: Perplexity value used in TSNE, default to 5.
    :param path_save_fig: Path of where to save the figure when generated. If none, it does not save the figure

    """
    tsne = TSNE(perplexity=perplexity, metric='cosine', init='pca', learning_rate='auto', square_distances=True,
                random_state=42)

    tsne_results = tsne.fit_transform(np.vstack([prototypes, objects]))
    tsne_protos = tsne_results[:len(prototypes)]
    tsne_embeds = tsne_results[len(prototypes):]

    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], s=10, alpha=0.6, c='#74add1', label=object_legend_text)
    plt.scatter(tsne_protos[:, 0], tsne_protos[:, 1], s=30, c='#d73027', alpha=0.9, label='Prototypes')

    plt.axis('off')
    plt.tight_layout()
    plt.legend(loc="upper left", prop={'size': 13})
    if path_save_fig:
        plt.savefig(path_save_fig, format='pdf')
    plt.show()


def get_top_k_items(item_weights: np.ndarray, items_info: pd.DataFrame, proto_idx: int,
                    top_k: int = 10, invert: bool = False):
    """
    Used to generate the recommendations to a user prototype or find the closest items to an item prototypes (depending
    on what item_weights encodes). In the ProtoMF paper, we use the **item-to-item-prototype similarity matrix** as
    item_weights when interpreting the item prototypes. We use the **list of all item embeddings** as item_weights when
    interpreting the user prototypes (this corresponds in finding the recommendations for a user which is maximally
    close to a specific user prototype a maximally distant from all the others).
    :param item_weights: Vector having, for each item, a value for each prototype. Shape is (n_items, n_prototypes)
    :param items_info: a dataframe which contains the item_id field used to look up the item information
    :param proto_idx: index of the prototype
    :param top_k: number of items to return for the prototype, default to 10
    :param invert: whether to look for the farthest items instead of closest, default to false
    :return: a DataFrame containing the top-k closest items to the prototype along with an item weight field.
    """
    assert proto_idx < item_weights.shape[1], \
        f'proto_idx {proto_idx} is too high compared to the number of available prototype'

    weights_proto = item_weights[:, proto_idx]

    top_k_indexes = np.argsort(weights_proto if invert else -weights_proto)[:top_k]
    top_k_weights = weights_proto[top_k_indexes]

    item_infos_top_k = items_info.set_index('item_id').loc[top_k_indexes]
    item_infos_top_k['item weight'] = top_k_weights
    return item_infos_top_k


def weight_visualization(u_sim_mtx: np.ndarray, u_proj: np.ndarray, i_sim_mtx: np.ndarray, i_proj: np.ndarray,
                         annotate_top_k: int = 3):
    """
    Creates weight visualization plots which is used to explain the recommendation of ProtoMF
    :param u_sim_mtx,...,i_proj: vectors that are obtained by the UI-PROTOMF model given the user and item pair.
    :param annotate_top_k: how many of the highest logits need to be annotated
    """

    rescale = lambda y: 1 - ((y + np.max(y)) / (np.max(y) * 2))

    def compute_ylims(array):
        y_lim_max = np.max(array) * (1 + 1 / 9)
        y_lim_min = np.min(array) * (1 + 1 / 9)
        return y_lim_min, y_lim_max

    # Computing the logits

    u_prods = u_sim_mtx * i_proj
    i_prods = i_sim_mtx * u_proj

    u_dot = u_prods.sum()
    i_dot = i_prods.sum()

    i_n_prototypes = i_sim_mtx.shape[-1]
    u_n_prototypes = u_sim_mtx.shape[-1]

    # Rescale the plots according to the number of prototypes
    i_vis_ratio = i_n_prototypes / (i_n_prototypes + u_n_prototypes)
    u_vis_ratio = 1 - i_vis_ratio

    # Compute max and mins of the visualization of the logits
    prods_lims = compute_ylims(np.concatenate([u_prods, i_prods]))
    proj_lims = compute_ylims(np.concatenate([u_proj, i_proj]))
    sim_mtx_lims = (0, compute_ylims(np.concatenate([u_sim_mtx, i_sim_mtx]))[1])

    # Plotting the users
    u_fig, u_axes = plt.subplots(3, 1, sharey='row', dpi=100, figsize=(8 * u_vis_ratio, 8))
    u_x = np.arange(u_n_prototypes)

    bars_u_prods = u_axes[0].bar(u_x, u_prods, color=plt.get_cmap('coolwarm')(rescale(u_prods)))
    bars_i_proj = u_axes[1].bar(u_x, i_proj, color=plt.get_cmap('coolwarm')(rescale(i_proj)))
    bars_u_sim_mtx = u_axes[2].bar(u_x, u_sim_mtx, color=plt.get_cmap('coolwarm')(rescale(u_sim_mtx)))

    u_axes[0].set_ylim(prods_lims)
    u_axes[1].set_ylim(proj_lims)
    u_axes[2].set_ylim(sim_mtx_lims)

    u_annotate_protos = np.argsort(-u_prods)[:annotate_top_k]
    for idx, bars in enumerate([bars_u_prods, bars_i_proj, bars_u_sim_mtx]):
        for u_annotate_idx in u_annotate_protos:
            bar = bars[u_annotate_idx]
            label_x = bar.get_x() - 0.8
            label_y = bar.get_height() + (2e-2 if idx == 2 else 1e-2)
            u_axes[idx].annotate(f'{u_annotate_idx}', (label_x, label_y), fontsize=11)

    u_axes[0].set_xlabel(r'$ {\mathbf{s}}^{\mathrm{user}}$', fontsize=24)
    u_axes[1].set_xlabel('$ \hat{\mathbf{t}} $', fontsize=24)
    u_axes[2].set_xlabel('$ \mathbf{u}^{*} $', fontsize=24)
    plt.tight_layout()
    plt.plot()

    # Plotting the items
    i_fig, i_axes = plt.subplots(3, 1, sharey='row', dpi=100, figsize=(i_vis_ratio * 8, 8))
    i_x = np.arange(i_n_prototypes)

    bars_i_prods = i_axes[0].bar(i_x, i_prods, color=plt.get_cmap('coolwarm')(rescale(i_prods)))
    bars_u_proj = i_axes[1].bar(i_x, u_proj, color=plt.get_cmap('coolwarm')(rescale(u_proj)))
    bars_i_sim_mtx = i_axes[2].bar(i_x, i_sim_mtx, color=plt.get_cmap('coolwarm')(rescale(i_sim_mtx)))

    i_axes[0].set_ylim(prods_lims)
    i_axes[1].set_ylim(proj_lims)
    i_axes[2].set_ylim(sim_mtx_lims)

    # Annotations
    i_annotate_protos = np.argsort(-i_prods)[:annotate_top_k]
    for idx, bars in enumerate([bars_i_prods, bars_u_proj, bars_i_sim_mtx]):
        for i_annotate_idx in i_annotate_protos:
            bar = bars[i_annotate_idx]
            label_x = bar.get_x() + (-0.8 if idx == 2 else +0)
            label_y = bar.get_height() + (2e-2 if idx == 2 else 1e-2)
            i_axes[idx].annotate(f'{i_annotate_idx}', (label_x, label_y), fontsize=11)

    i_axes[0].set_xlabel('$ \mathbf{s}^{\mathrm{item}} $', fontsize=24)
    i_axes[1].set_xlabel('$ \hat{\mathbf{u}} $', fontsize=24)
    i_axes[2].set_xlabel('$ \mathbf{t}^{*} $', fontsize=24)
    plt.tight_layout()
    plt.plot()
