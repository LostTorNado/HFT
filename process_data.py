import tarfile
import os
import pandas as pd
import json
from tqdm import tqdm
from multiprocess import Pool

fpath = 'E:/BaiduNetdiskDownload/观澜行情采集'
fnames = os.listdir(fpath)
local_path = 'E:/ryse/LocalDatabase'
code = 'IF'
os.makedirs(f'data/{code}', exist_ok=True)
f = pd.read_csv(local_path + '/' + code + '.csv')
f = f[f['Date'] >= 20230302].reset_index(drop=True)
# for dt in tqdm(range(len(f['Date']))):

def main(dt):
    date = f['Date'][dt]
    filename = f'{code}_{date}.csv'

    if os.path.exists(f'data/{code}/{filename}'):
        return
    if date == 20230829:
        return
    tarfilename = f'md-{date}.tar.gz'
    if tarfilename not in fnames:
        return
    maincode = f['FCodeHisCode'][dt].split('.')[0]
    with tarfile.open(fpath + '/' + tarfilename, 'r:gz') as tar:
        member = tar.getmember(f'{f['Date'][dt]}/{maincode}_{f['Date'][dt]}.txt')
        with tar.extractfile(member) as ff:
            txt_data = ff.read().decode('utf-8')
    #         data = [json.loads(i) for i in txt_data.split('\n')]
    data = pd.DataFrame([json.loads(line) for line in txt_data.strip().split('\n')])
    #         data = pd.DataFrame(data)
    data.to_csv(f'data/{code}/{filename}', index=False)

if __name__ == '__main__':
    with Pool(processes = 4) as pool:
        result = pool.map(lambda atsl:main(atsl), range(len(f['Date'])))