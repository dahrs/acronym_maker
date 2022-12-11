#!/usr/bin/python
# -*- coding:utf-8 -*-

import re
import os
import json
import edlib
import epitran
import argparse
from tqdm import tqdm
from os.path import isfile
from collections import deque
from datetime import datetime
from multiprocessing import Pool
from nltk.corpus import stopwords
from itertools import permutations
from itertools import combinations
from nltk.tokenize import word_tokenize


def change_ind_with_initial(an_iter: tuple, ind: int, initials_list) -> list[tuple]:
    an_iter = list(an_iter)
    return [tuple(an_iter[:ind] + [init] + an_iter[ind+1:]) for init in initials_list]


def has_int(an_iter: [list, tuple]) -> [bool, int]:
    for ind_elem, elem in enumerate(an_iter):
        if type(elem) is int:
            return ind_elem
    return False


def ind_to_initials(iter_and_init_l: tuple[tuple, list]):
    an_iter, all_initials = iter_and_init_l
    clean_ones = []
    q = deque()
    q.append(an_iter)
    while q:
        a_perm = q.popleft()
        a_int = has_int(a_perm)
        if a_int is not False:
            changes_perms = change_ind_with_initial(a_perm, a_int, all_initials[a_perm[a_int]])
            for cp in changes_perms:
                q.append(cp)
        else:
            clean_ones.append(a_perm)
    return clean_ones


def get_candidates(word_initials_dict: dict, nb_process: int = 10) -> set[str]:
    # arrange for combinations
    combis = []
    initials = list(word_initials_dict.values())
    combi = range(0, len(word_initials_dict))
    for size in range(len(combi)):
        for co in combinations(combi, size + 1):
            # combis += [(perm, initials) for perm in permutations(co)]
            # TODO: move the permutations to the analysis part
            combis.append((co, initials))
    # get the candidates in all their variants
    candidates = set()
    with Pool(processes=nb_process) as pool:
        for clean_str_l in tqdm(pool.imap_unordered(ind_to_initials, combis)):
            for clean_str in clean_str_l:
                candidates.add(clean_str)
    return candidates


def has_needed_initials(tok_s: str, cand_li: list[str]) -> bool:
    for ccl in cand_li:
        if ccl[0] not in tok_s:
            return False
    return True


def compare_with_toks(init_and_toks: list[tuple[str, any]]) -> [tuple, None]:
    init_l, dict_toks = init_and_toks
    best = None
    if len(init_l) == 1:
        return None
    dict_toks = [dt for dt in dict_toks if len(init_l) - 2 < len(dt) < len(init_l) + 2]
    for c_l in permutations(init_l):
        word = "".join(c_l)
        for d_tok in dict_toks:
            if has_needed_initials(d_tok, c_l):
                edit_dist = edlib.align(word, d_tok)["editDistance"]
                if not best:
                    best = (edit_dist, d_tok, c_l)
                elif edit_dist <= best[0]:
                    best = (edit_dist, d_tok, c_l)
    return best


def dump_final_tsv(rank_l: list[tuple], out_path: str, reverse_dict: dict):
    fin_tsv = ["Acronym suggestion\tWords used\n"]
    for rrnk in rank_l:
        fin_tsv.append(f"{rrnk[1].upper()}\t{' ; '.join([' OR '.join(reverse_dict[ini]) for ini in rrnk[2]])}\n")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.writelines(fin_tsv)


def main(input_json: str = "./input.json",
         output_perms: str = "./tmp.json",
         output_tsv: str = "./output.tsv",
         nb_processes: int = 10,
         lang: str = "en") -> None:
    with open(input_json, encoding="utf-8") as in_f:
        in_d = json.load(in_f)
    # unify and complete the dict
    in_l_d = {}
    rev_in_l_d = {}
    for kk, vv in in_d.items():
        kk = kk.lower()
        in_l_d[kk] = set([vel.lower() for vel in vv])
        # add initial if it's not amongst the choices
        toks = word_tokenize(kk, language="english")
        toks_ns = [tk for tk in toks if tk not in stopwords.words("english")]
        for words in [toks, toks_ns]:
            initial_1 = ""
            # add first couple of letters instead of the first initial
            initial_2 = ""
            initial_3 = ""
            for wrd in words:
                initial_1 = f"{initial_1}{wrd[0]}"
                initial_2 = f"{initial_2}{wrd[:2]}"
                initial_3 = f"{initial_3}{wrd[:3]}"
            for initial in [initial_1, initial_2, initial_3]:
                if initial not in in_l_d[kk]:
                    in_l_d[kk].add(initial)
        # populate the reverse dict
        for v_el in in_l_d[kk]:
            if v_el not in rev_in_l_d:
                rev_in_l_d[v_el] = []
            rev_in_l_d[v_el].append(kk)
    # make a list of all permutations of the initials we do have
    if not isfile(output_perms):
        candidates = get_candidates(in_l_d, nb_processes)
        with open(output_perms, mode="w", encoding="utf-8") as out_f:
            json.dump(list(candidates), out_f)
    else:
        with open(output_perms, encoding="utf-8") as in_f:
            candidates = json.load(in_f)
    # browse the whole wikipedia resource to get the best 100 candidates
    with open(f"./resources/{lang}Tok1000.json", encoding="utf-8") as lang_f:
        lang_toks = json.load(lang_f)
    rank100 = []
    cand_plus_dict = [(cand, lang_toks) for cand in candidates]
    last_time = datetime.now().timestamp()
    with Pool(processes=nb_processes) as pool:
        # measure closeness by levenshtein
        for best in tqdm(pool.imap_unordered(compare_with_toks, cand_plus_dict)):
            if best:
                rank100.append(best)
                rank100 = sorted(rank100, key=lambda x: x[0], reverse=False)
                # dump save every 10 min
                if datetime.now().timestamp() - last_time > 600:
                    dump_final_tsv(rank100, output_tsv, rev_in_l_d)
                    last_time = datetime.now().timestamp()
    # output the final document
    dump_final_tsv(rank100, output_tsv, rev_in_l_d)
    #  remove the temporary file
    os.remove(output_perms)
    # # TODO: incorporate sound similarity with epitran and the tok 2 ipa dicts
    # with open(f"./resources/{lang}_ipa.json", encoding="utf-8") as ipa_f:
    #     ipa_toks = json.load(ipa_f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slow but efficient acronym maker for catchy academic papers.")

    parser.add_argument("-in", "--input", type=str, default="./input.json",
                        help="Path to the input json file where the words and initials to make the acronym appear.")
    parser.add_argument("-tmp", "--temporary", type=str, default="./tmp.json",
                        help="Path to the temporary json file where to save the first wave of initials combinations.")
    parser.add_argument("-out", "--output", type=str, default="./output.tsv",
                        help="Path to the output tsv (tab separated values) file where the end file will be saved.")
    parser.add_argument("-p", "--processes", type=int, default=10,
                        help="Amount of processes to launch simultaneously in order to parallelize/thread.")
    parser.add_argument("-lang", "--language", type=str, default="en",
                        help="Language, writen in the ISO 639-1:2002 2-character code, in lowercase.")
    args = parser.parse_args()

    main(args.input, args.temporary, args.output, args.processes, args.lang)
