import os

def findAllFilesWithSpecifiedName(target_dir, target_name):
    find_res = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            if target_name in file:
                find_res.append(os.path.join(root_path, file))
    find_res.sort()
    return find_res