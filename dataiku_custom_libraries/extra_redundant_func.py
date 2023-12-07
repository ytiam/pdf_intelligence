def extract_id(s):
    id_ = []
    for i in s:
        if i.isdigit():
            id_.append(i)
        else:
            break
    id__ = ''.join(id_)
    return id__

def alpha_to_num(s):
    numbers = []
    s = ''.join(s.split())
    for s_ in s:
        number = ord(s_) - 96
        numbers.append(str(number))
    return ''.join(numbers)

def new_id_generate_(file_name):
    ids = dss_read_pickle_from_folder("misc","doc_ids.pickle")
    new_id = random.randint(10**8, 10**9)
    while str(new_id) in ids:
        new_id = random.randint(10**8, 10**9)
    ids = ids.append({"doc":file_name,"doc_id_parsed":str(new_id)},ignore_index=True) 
    dss_write_pickle_to_folder(ids,"misc","doc_ids.pickle")
    return new_id

def validate_generate_doc_id(file_name_tuple):
    k = file_name_tuple[0]
    v = file_name_tuple[1]
    if v.isalpha():
        id_ = new_id_generate(k)
    else:
        id_ = k
    return id_