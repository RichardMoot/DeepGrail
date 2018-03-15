
import pickle

def save_all_bis():
    save_obj('word_to_index', word_to_index)
    save_obj('index_to_word', index_to_word)
    save_obj('super_to_index', pos1_to_index)
    save_obj('index_to_super', index_to_pos1)
    save_obj('word_to_vec_map', word_to_vec_map)
    save_obj('p1_to_integer', p1_to_integer)
    save_obj('integer_to_p1', integer_to_p1)
    save_obj('p2_to_integer', p2_to_integer)
    save_obj('integer_to_p2', integer_to_p2)
    save_obj('p3_to_integer', p3_to_integer)
    save_obj('integer_to_p3', integer_to_p3)
    save_obj('p4_to_integer', p4_to_integer)
    save_obj('integer_to_p4', integer_to_p4)
    save_obj('s1_to_integer', s1_to_integer)
    save_obj('integer_to_s1', integer_to_s1)
    save_obj('s2_to_integer', s2_to_integer)
    save_obj('integer_to_s2', integer_to_s2)
    save_obj('s3_to_integer', s3_to_integer)
    save_obj('integer_to_s3', integer_to_s3)
    save_obj('s4_to_integer', s4_to_integer)
    save_obj('integer_to_s4', integer_to_s4)
    save_obj('s5_to_integer', s5_to_integer)
    save_obj('integer_to_s5', integer_to_s5)
    save_obj('s6_to_integer', s6_to_integer)
    save_obj('integer_to_s6', integer_to_s6)
    save_obj('s7_to_integer', s7_to_integer)
    save_obj('integer_to_s7', integer_to_s7)


def save_obj(name, obj):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
