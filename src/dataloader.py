import os, csv, json

DIAG_SPLIT = 0.2

def load_simp_eval(path, limit=None, split='main'):
    """
    Load SimpEval
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = [dict(zip(header, row)) for row in csv_reader]

    # Exclude non-human systems
    data = [x for x in data if 'Human' in x['system']]
    if limit: data = data[:limit]

    if split == 'diagnostic':
        data = data[:DIAG_SPLIT*len(data)]  
    elif split == 'main':
        data = data[DIAG_SPLIT*len(data):]
    else:
        raise ValueError(split)

    # Rename the human "generation" as a reference
    for i, s in enumerate(data):
        s['references'] = [s['generation']]
        s['source'] = s['original']
        s['metadata'] = { '_id': i }

    return data


def package_translation(source_data, reference_data, limit=None, sorted_by_length=False, split='main'):
    """
    Given loaded lists of translations, package into the desired data format
    """
    data = []
    for i, s in enumerate(source_data):
        data += [{
            'source': s,
            'references': reference_data[i]
        }]

    if sorted_by_length:
        print('Using sorted data...')
        data = sorted(data, key=lambda x: len(x['source']), reverse=True)

    if limit: 
        data = data[:limit]

    if split == 'diagnostic':
        data = data[:DIAG_SPLIT*len(data)]  
    elif split == 'main':
        data = data[DIAG_SPLIT*len(data):]
    else:
        raise ValueError(split)

    for i, s in enumerate(data):
        s['metadata'] = { '_id': i }

    return data


def load_newstest19(path, src_lang, tgt_lang, limit=None, sorted_by_length=False, split='main'):
    """
    Load WMT 2019 News Test data
    """
    with open(os.path.join(path, f'newstest2019-{src_lang}{tgt_lang}-src.{src_lang}.sgm'), 'r', encoding='utf-8') as f:
        source_data = [''.join(l.split('">')[1:]).split("</seg>")[0] for l in f.read().splitlines() if l.strip().startswith("<seg id")]

    with open(os.path.join(path, f'newstest2019-{src_lang}{tgt_lang}-ref.{tgt_lang}.sgm'), 'r', encoding='utf-8') as f:
        reference_data = [''.join(l.split('">')[1:]).split("</seg>")[0] for l in f.read().splitlines() if l.strip().startswith("<seg id")]
        reference_data = [[ref] for ref in reference_data]

    # (Optional) Load additional reference data
    optional_ref_path = os.path.join(path, f'newstest2019-{src_lang}{tgt_lang}-ref2.{tgt_lang}')
    if os.path.exists(optional_ref_path):
        with open(optional_ref_path, 'r', encoding='utf-8') as f:
            additional_reference_data = [l.strip() for l in f.readlines()]
            assert len(additional_reference_data) == len(reference_data)
            for i, _ in enumerate(reference_data):
                reference_data[i] += [additional_reference_data[i]]
        
    return package_translation(source_data, reference_data, limit=limit, sorted_by_length=sorted_by_length, split=split)


def load_alma_test(path, src_lang, tgt_lang, limit=None, sorted_by_length=False, split='main'):
    """
    Load ALMA test data
    """
    not_en = src_lang if src_lang != 'en' else tgt_lang
    with open(os.path.join(path, f'{not_en}en', f'test.{src_lang}-{tgt_lang}.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        source_data = [l['translation'][src_lang] for l in data]
        reference_data = [[l['translation'][tgt_lang]] for l in data]
        
    return package_translation(source_data, reference_data, limit=limit, sorted_by_length=sorted_by_length, split=split)


def load_ntrex(path, src_lang, tgt_lang, limit=None, sorted_by_length=False, split='main'):
    """
    Load NTREX translation data
    """
    if src_lang == 'en': src_lang = 'eng'
    with open(os.path.join(path, f'newstest2019-src.{src_lang}.txt'), 'r', encoding='utf-8') as f:
        source_data = [l for l in f.read().splitlines()]

    with open(os.path.join(path, f'newstest2019-ref.{tgt_lang}.txt'), 'r', encoding='utf-8') as f:
        reference_data = [[l] for l in f.read().splitlines()]
        
    return package_translation(source_data, reference_data, limit=limit, sorted_by_length=sorted_by_length, split=split)


def load_human_eval(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(l) for l in f]
    data = []
    for i, entry in enumerate(raw_data):
        data += [{
            'source': entry['prompt'],
            'references': entry['canonical_solution'],
            'metadata': {
                '_id': i,
                'task_id': entry['task_id'],
                'entry_point': entry['entry_point'],
                'test': entry['test']
            }
        }]
    if limit: data = data[:limit]
    return data


def load_prompts(prompt_file, return_weights=False):
    """
    Load custom prompts JSON file
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    if isinstance(prompts[0], str):
        return prompts
    elif isinstance(prompts[0], dict):
        if 'task_id' in prompts[0].keys():
            print("Found HumanEval prompts, returning with no processing...")
            return prompts

        if return_weights:
            return [p['prompt'] for p in prompts], [p['weight'] for p in prompts]
        else:
            return [p['prompt'] for p in prompts]
    else:
        raise ValueError(f'Cannot parse prompt file: {prompts}')