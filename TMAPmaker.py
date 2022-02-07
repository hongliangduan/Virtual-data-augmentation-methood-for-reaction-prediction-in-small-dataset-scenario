import numpy as np
from faerun import Faerun
import tmap as tm
from matplotlib.colors import ListedColormap

def map_makers(R_SIZE,names,R_SMILES,R_FPS):

    # Use 256 permutations to create the MinHash
    print('make map')
    enc = tm.Minhash(256)
    lf = tm.LSHForest(256)
    LABELS = []
    for i, j in enumerate(R_SIZE): ##生成LABEL值
        LABELS=LABELS+[i]*j
    legend_labels = set()
    for i, j in enumerate(names):
        legend_labels.update({(i,j[0])})

    test_LABELS=[R_SMILES[i]
                 +'__'
                 # +str(i)
                 +'__'
                 # +str(i)
                 +'__'
                 # +str(i)
                 +'__'
                 +R_SMILES[i]
                 +'__' for i in range(len(R_SMILES))]
    # test_LABELS=[test_LABELS[n].replace(">", "to") for n in range(len(test_LABELS))]

    lf.batch_add(enc.batch_from_weight_array(R_FPS,'I2CWS'))

    # Index the added data
    lf.index()
    custom_cmap = ListedColormap(
        ["#94B7CF","#BBE1E3","#A4CAB6","#DFECD5","#F0A1A8","#F7CED2","#F9EEBD","#F1D79D","#D1C2D3","#E8D7DF"],
        name="custom",
    )
    print('plot scatter')
    # Construct the k-nearest neighbour graph
    CFG=tm.LayoutConfiguration()
    CFG.k = 50
    CFG.sl_extra_scaling_steps = 20
    CFG.node_size = 1/20
    x,y,s,t,_=tm.layout_from_lsh_forest(lf,CFG)

    faerun = Faerun(view="front",coords=False,clear_color="#FFFFFF", legend_title="")
    faerun.add_scatter(
        "REACTION",
        {"x": x, "y": y, "c": LABELS, "labels": test_LABELS},
        colormap=custom_cmap,
        point_scale=5.0,
        max_point_size=10,
        shader="smoothCircle",
        # shader="sphere",
        has_legend=True,
        categorical=True,
        legend_labels=legend_labels,
        legend_title="reaction"
    )
    faerun.add_tree(
        "REACTION_tree", {"from": s, "to": t}, point_helper="REACTION", color="#666666"
    )
    faerun.plot("reaction",template="reaction_smiles")