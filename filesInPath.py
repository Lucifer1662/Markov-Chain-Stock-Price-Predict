from os import walk

def filesInPath(path : str):
    fileNameFromPath = []
    for (dirpath, dirnames, filenames) in walk(path):
        fileNameFromPath.extend(filenames)
        break
    return fileNameFromPath