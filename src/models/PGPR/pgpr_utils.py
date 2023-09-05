from __future__ import absolute_import, division, print_function
from easydict import EasyDict as edict

import os
import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import csv
#import scipy.sparse as sp
import torch
from collections import defaultdict

# Dataset names.
#from sklearn.feature_extraction.text import TfidfTransformer

ML1M = 'ml1m'
LFM1M = 'lfm'
CELL = 'cellphones'

# Dataset directories.
DATASET_DIR = {
    LFM1M: f'../../data/{LFM1M}/preprocessed/pgpr',
}

# Model result directories.
TMP_DIR = {
    LFM1M: f'{DATASET_DIR[LFM1M]}/tmp',
}

# Label files.
LABELS = {
    LFM1M: (TMP_DIR[LFM1M] + '/train_label.pkl', TMP_DIR[LFM1M] + '/test_label.pkl'),
}


#LASTFM ENTITIES
MICRO_GENRE = 'micro_genre'
ARTIST = 'artist'
ALBUM = 'album'
GENRE = 'genre'


#SHARED ENTITIES
USER = 'user'
PRODUCT = 'product'


#LASTFM RELATIONS
LISTENED_TO = 'listened_to'
IN_ALBUM = 'in_album'
HAS_MICRO_GENRE = 'has_micro_genre'
CREATED_BY = 'created_by'
HAS_GENRE = 'has_genre'

SELF_LOOP = 'self_loop'

RELATION_IDS = {
    LISTENED_TO: 0,
    HAS_MICRO_GENRE: 1,
    IN_ALBUM: 2,
    HAS_GENRE: 3,
    CREATED_BY: 4,
    SELF_LOOP: 5
}


KG_RELATION = {
    LFM1M: {
        USER: {
            LISTENED_TO: PRODUCT,
        },
        ARTIST: {
            CREATED_BY: PRODUCT,
        },
        PRODUCT: {
            LISTENED_TO: USER,
            CREATED_BY: ARTIST,
            HAS_MICRO_GENRE: MICRO_GENRE,
            HAS_GENRE: GENRE,
            IN_ALBUM: ALBUM
        },
        MICRO_GENRE: {
            HAS_MICRO_GENRE: PRODUCT,
        },
        GENRE: {
            HAS_GENRE: PRODUCT,
        },
        ALBUM: {
            IN_ALBUM: PRODUCT,
        },
    }
}


#0 is reserved to the main relation, 1 to mention
PATH_PATTERN = {
    LFM1M: {
        0: ((None, USER), (LISTENED_TO, PRODUCT), (LISTENED_TO, USER), (LISTENED_TO, PRODUCT)),
        2: ((None, USER), (LISTENED_TO, PRODUCT), (HAS_GENRE, GENRE), (HAS_GENRE, PRODUCT)),
        3: ((None, USER), (LISTENED_TO, PRODUCT), (CREATED_BY, ARTIST), (CREATED_BY, PRODUCT)),
        4: ((None, USER), (LISTENED_TO, PRODUCT), (IN_ALBUM, ALBUM), (IN_ALBUM, PRODUCT)),
        5: ((None, USER), (LISTENED_TO, PRODUCT), (HAS_MICRO_GENRE, MICRO_GENRE), (HAS_MICRO_GENRE, PRODUCT)),
        6: ((None, USER), (LISTENED_TO, PRODUCT), (CREATED_BY, ARTIST), (CREATED_BY, PRODUCT), (HAS_GENRE, GENRE), (HAS_GENRE, PRODUCT)),
        #10: ((None, USER), (LISTENED, PRODUCT), (FEATURED_BY, FEATURED_ARTIST), (FEATURED_BY, PRODUCT)),
    }
}


MAIN_PRODUCT_INTERACTION = {
    LFM1M: (PRODUCT, LISTENED_TO),
}



def get_entities(dataset_name):
    return list(KG_RELATION[dataset_name].keys())


def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans


def get_dataset_relations(dataset_name, entity_head):
    return list(KG_RELATION[dataset_name][entity_head].keys())


def get_entity_tail(dataset_name, relation):
    entity_head, _ = MAIN_PRODUCT_INTERACTION[dataset_name]
    return KG_RELATION[dataset_name][entity_head][relation]


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
        # CHANGED
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed

#Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]



def get_pid_to_kgid_mapping(dataset_name):
    if dataset_name == "ml1m":
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/movie.txt", "r")
    elif dataset_name == LFM1M:
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/song.txt", "r")
    else:
        print("Dataset mapping not found!")
        exit(-1)
    reader = csv.reader(file, delimiter=' ')
    dataset_pid2kg_pid = {}
    next(reader, None)
    for row in reader:
        if dataset_name == "ml1m" or dataset_name == LFM1M:
            dataset_pid2kg_pid[int(row[0])] = int(row[1])
    file.close()
    return dataset_pid2kg_pid


def get_entity_edict(dataset_name):
    if dataset_name == ML1M:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            actor='actor.txt.gz',
            composer='composer.txt.gz',
            director='director.txt.gz',
            producer='producer.txt.gz',
            production_company='production_company.txt.gz',
            category='category.txt.gz',
            country='country.txt.gz',
            editor='editor.txt.gz',
            writter='writter.txt.gz',
            cinematographer='cinematographer.txt.gz',
            wikipage='wikipage.txt.gz',
        )
    elif dataset_name == LFM1M:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            artist='artist.txt.gz',
            album='album.txt.gz',
            genre='genre.txt.gz',
            micro_genre='micro_genre.txt.gz',
        )
    elif dataset_name == CELL:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            related_product='related_product.txt.gz',
            brand='brand.txt.gz',
            category='category.txt.gz',
        )
    return entity_files


def get_validation_pids(dataset_name):
    if not os.path.isfile(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')):
        return []
    validation_pids = defaultdict(set)
    with open(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')) as valid_file:
        reader = csv.reader(valid_file, delimiter=" ")
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            validation_pids[uid].add(pid)
    valid_file.close()
    return validation_pids

def get_uid_to_kgid_mapping(dataset_name):
    dataset_uid2kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/entities/mappings/user.txt", 'r') as file:
        reader = csv.reader(file, delimiter=" ")
        next(reader, None)
        for row in reader:
            if dataset_name == "ml1m" or dataset_name == LFM1M:
                uid_review = int(row[0])
            uid_kg = int(row[1])
            dataset_uid2kg_uid[uid_review] = uid_kg
    return dataset_uid2kg_uid

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr