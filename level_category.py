import pandas as pd
import numpy as np
import pyabf
from sklearn.cluster import KMeans

ref_base_lvl_map = {
    "AAAA": 45.8,
    "AAAC": 42.1,
    "AAAG": 47.5,
    "AAAT": 46.4,
    "AACA": 46.4,
    "AACC": 39.3,
    "AACG": 41.6,
    "AACT": 39.7,
    "AAGA": 47.5,
    "AAGC": 41.5,
    "AAGG": 45.2,
    "AAGT": 44.4,
    "AATA": 40.2,
    "AATC": 35.2,
    "AATG": 39.4,
    "AATT": 36.7,
    "ACAA": 46.4,
    "ACAC": 43.4,
    "ACAG": 44.7,
    "ACAT": 45.2,
    "ACCA": 42.7,
    "ACCC": 40.7,
    "ACCG": 46.3,
    "ACCT": 41.5,
    "ACGA": 46.6,
    "ACGC": 39.7,
    "ACGG": 44.5,
    "ACGT": 42.3,
    "ACTA": 41.9,
    "ACTC": 37.4,
    "ACTG": 40.1,
    "ACTT": 36.1,
    "AGAA": 56.3,
    "AGAC": 45.0,
    "AGAG": 54.1,
    "AGAT": 47.7,
    "AGCA": 49.1,
    "AGCC": 41.8,
    "AGCG": 51.7,
    "AGCT": 44.0,
    "AGGA": 47.9,
    "AGGC": 38.3,
    "AGGG": 41.8,
    "AGGT": 37.4,
    "AGTA": 31.5,
    "AGTC": 24.3,
    "AGTG": 34.6,
    "AGTT": 27.3,
    "ATAA": 44.6,
    "ATAC": 42.5,
    "ATAG": 42.0,
    "ATAT": 42.8,
    "ATCA": 37.2,
    "ATCC": 35.3,
    "ATCG": 39.8,
    "ATCT": 37.4,
    "ATGA": 36.4,
    "ATGC": 30.7,
    "ATGG": 36.0,
    "ATGT": 33.0,
    "ATTA": 31.6,
    "ATTC": 26.4,
    "ATTG": 31.4,
    "CAAA": 46.4,
    "CAAC": 43.5,
    "CAAG": 47.5,
    "CAAT": 43.2,
    "CACA": 47.1,
    "CACC": 40.6,
    "CACG": 39.7,
    "CACT": 41.1,
    "CAGA": 47.7,
    "CAGC": 37.0,
    "CAGG": 42.9,
    "CAGT": 37.4,
    "CATA": 42.8,
    "CATC": 35.3,
    "CATG": 38.9,
    "CATT": 31.4,
    "CCAA": 45.3,
    "CCAC": 42.7,
    "CCAG": 43.7,
    "CCAT": 43.3,
    "CCCA": 46.3,
    "CCCC": 42.6,
    "CCCG": 43.0,
    "CCCT": 37.8,
    "CCGA": 46.3,
    "CCGC": 40.7,
    "CCGG": 43.2,
    "CCGT": 38.0,
    "CCTA": 43.9,
    "CCTC": 37.8,
    "CCTG": 34.1,
    "CCTT": 32.8,
    "CGAA": 51.3,
    "CGAC": 42.4,
    "CGAG": 46.4,
    "CGAT": 44.4,
    "CGCA": 47.8,
    "CGCC": 40.2,
    "CGCG": 43.1,
    "CGCT": 40.7,
    "CGGA": 40.7,
    "CGGC": 33.5,
    "CGGG": 36.2,
    "CGGT": 31.6,
    "CGTA": 27.0,
    "CGTC": 16.8,
    "CGTG": 21.8,
    "CGTT": 17.7,
    "CTAA": 47.3,
    "CTAC": 40.4,
    "CTAG": 44.8,
    "CTAT": 37.0,
    "CTCA": 44.1,
    "CTCC": 37.5,
    "CTCG": 35.7,
    "CTCT": 38.6,
    "CTGA": 34.3,
    "CTGC": 27.5,
    "CTGG": 34.1,
    "CTGT": 24.5,
    "CTTA": 31.2,
    "CTTC": 22.1,
    "CTTG": 27.4,
    "CTTT": 24.0,
    "GAAA": 56.3,
    "GAAC": 44.6,
    "GAAG": 48.7,
    "GAAT": 48.6,
    "GACA": 45.8,
    "GACC": 38.7,
    "GACG": 42.2,
    "GACT": 39.7,
    "GAGA": 47.4,
    "GAGC": 43.3,
    "GAGG": 43.4,
    "GAGT": 45.4,
    "GATA": 41.0,
    "GATC": 33.9,
    "GATG": 39.4,
    "GATT": 35.6,
    "GCAA": 49.1,
    "GCAC": 38.7,
    "GCAG": 48.0,
    "GCAT": 38.9,
    "GCCA": 44.4,
    "GCCC": 41.5,
    "GCCG": 43.2,
    "GCCT": 41.3,
    "GCGA": 47.2,
    "GCGC": 38.6,
    "GCGG": 45.6,
    "GCGT": 37.2,
    "GCTA": 36.8,
    "GCTC": 33.7,
    "GCTG": 36.0,
    "GCTT": 31.6,
    "GGAA": 51.8,
    "GGAC": 41.8,
    "GGAG": 46.7,
    "GGAT": 44.1,
    "GGCA": 43.7,
    "GGCC": 40.0,
    "GGCG": 42.3,
    "GGCT": 37.9,
    "GGGA": 41.8,
    "GGGC": 31.3,
    "GGGG": 30.2,
    "GGGT": 30.7,
    "GGTA": 28.0,
    "GGTC": 17.8,
    "GGTG": 25.5,
    "GGTT": 20.3,
    "GTAA": 40.3,
    "GTAC": 38.5,
    "GTAG": 38.0,
    "GTAT": 37.4,
    "GTCA": 35.7,
    "GTCC": 27.4,
    "GTCG": 30.6,
    "GTCT": 28.5,
    "GTGA": 33.5,
    "GTGC": 24.4,
    "GTGG": 22.9,
    "GTGT": 26.9,
    "GTTA": 29.5,
    "GTTC": 21.2,
    "GTTG": 27.5,
    "GTTT": 22.2,
    "TAAA": 44.6,
    "TAAC": 42.5,
    "TAAG": 47.6,
    "TAAT": 50.5,
    "TACA": 44.4,
    "TACC": 42.5,
    "TACG": 42.3,
    "TACT": 43.2,
    "TAGA": 45.8,
    "TAGC": 41.3,
    "TAGG": 44.0,
    "TAGT": 41.8,
    "TATA": 40.7,
    "TATC": 36.1,
    "TATG": 37.9,
    "TATT": 35.0,
    "TCAA": 43.7,
    "TCAC": 39.7,
    "TCAG": 46.3,
    "TCAT": 48.1,
    "TCCA": 43.7,
    "TCCC": 38.6,
    "TCCG": 39.3,
    "TCCT": 41.2,
    "TCGA": 46.4,
    "TCGC": 40.4,
    "TCGG": 40.0,
    "TCGT": 37.9,
    "TCTA": 42.1,
    "TCTC": 34.6,
    "TCTG": 32.8,
    "TCTT": 32.4,
    "TGAA": 45.6,
    "TGAC": 38.5,
    "TGAG": 42.4,
    "TGAT": 43.3,
    "TGCA": 42.8,
    "TGCC": 35.4,
    "TGCG": 40.6,
    "TGCT": 37.3,
    "TGGA": 38.7,
    "TGGC": 32.1,
    "TGGG": 33.7,
    "TGGT": 27.1,
    "TGTA": 27.1,
    "TGTC": 17.6,
    "TGTG": 21.7,
    "TGTT": 18.8,
    "TTAA": 40.1,
    "TTAC": 35.6,
    "TTAG": 41.8,
    "TTAT": 40.6,
    "TTCA": 36.0,
    "TTCC": 28.7,
    "TTCG": 33.2,
    "TTCT": 32.4,
    "TTGA": 33.1,
    "TTGC": 27.3,
    "TTGG": 32.1,
    "TTGT": 28.8,
    "TTTA": 29.2,
    "TTTC": 22.2,
    "TTTG": 25.9,
    "TTTT": 22.2,
}

ref_base_lvl = pd.read_csv("/home/xjtu/Documents/DNA-sequencing/ref_base_lvl.csv")

seq_A='PAAAAAAACCTTCCXTTTTCCCGTCCGCTCGTTCGCGCCTGTCTGCTTGTTTGCGTGTGCCGGTCGGCTAAGCATTCTCATGCAGGTCGTAGCC'
seq_B='PAAAAAAACCTTCCXTGTTTGCGTGTGCCGGTCGGCTGGTTGGCGGGTGGGCCCATCAAAACACTCATAAGCATTCTCATGCAGGTCGTAGCC'

def seq2base(seq: str) -> list[str]:
    """
    将序列A/B转换为'4-碱基'序列
    1. 把 X 删去
    2. 从序列中提取长度为4的子序列
    3. 删去重复的子序列
    """
    seq = seq.replace('X', '')

    base_ls = []
    for i in range(1, len(seq) - 3):
        base_ls.append(seq[i:i+4])

    base_ls = list(set(base_ls))

    return base_ls


def abf2base(change_pts_lvl: pd.DataFrame, base_lvl_map: dict) -> pd.DataFrame:
    """
    通过change_pts_lvl和base_lvl_map生成电平序列对应的碱基序列
    """
    def find_base(level:float, base_lvl_map: dict) -> str:
        min_diff = float("inf")
        base = None
        for k, v in base_lvl_map.items():
            diff = abs(level - v)
            if diff < min_diff:
                min_diff = diff
                base = k
        return base

    # 在change_pts_lvl中加一列4-base
    change_pts_lvl["4-base"] = change_pts_lvl["level"].apply(lambda x: find_base(x, base_lvl_map))
    change_pts_lvl_base = change_pts_lvl
    return change_pts_lvl_base


def classify_level(change_pts_lvl: pd.DataFrame,seq:str) -> dict:
    """
    通过 KMeans 聚类算法对 level 进行分类
    实际分类数由 seq 决定, 因为 seq 中不包括全部 256 种 4-base
    1. seq->base_ls, 生成所有可能的 4-base, 得出种类数 num_of_cates
    2. 将电平分为 num_of_cates类, 
    3. base_ls 中的 4-base 依 ref_base_lvl_map 排序
    4. 根据电平大小顺序对应
    ->]
    change_pts_lvl:
    change_point,level
    519524,24.27726936340332
    519530,25.45401382446289
    ...
    [->
    base_lvl_map: {4-base: level}
    """
    base_ls = seq2base(seq)
    num_of_cates = len(base_ls)

    y = change_pts_lvl["level"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_of_cates, random_state=0).fit(y)
    arr_centers = np.sort(kmeans.cluster_centers_, axis=0)

    base_lvl = pd.DataFrame(columns=["4-base", "level"])
    base_lvl["4-base"] = base_ls
    base_lvl["level"] = base_lvl["4-base"].apply(lambda x: ref_base_lvl_map[x])
    base_lvl = base_lvl.sort_values(by='level')
    sorted_base_ls = base_lvl["4-base"]

    base_lvl_map = dict(zip(sorted_base_ls, arr_centers.flatten()))

    return base_lvl_map
