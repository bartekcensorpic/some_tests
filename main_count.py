from joblib import Parallel, delayed
import multiprocessing
from count_annotations import process_file
from pathlib import Path
import functools

if __name__ == '__main__':
    xml_path = r'C:\Users\barte\Documents\Projects\some_tests\files'
    output = r'C:\Users\barte\Desktop\trash\tmp\temp.txt'


    num_cores = multiprocessing.cpu_count()

    list_of_xml = Path(xml_path).glob('*.xml')

    results = Parallel(n_jobs=num_cores)(delayed(process_file)(file) for file in list_of_xml)
    debug = 5
    result = functools.reduce(lambda a,b : a+b, results)
    print(result)
    with open(output, 'w+') as f:
        f.write(str(result))