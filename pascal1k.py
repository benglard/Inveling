from bs4 import BeautifulSoup as soup
from requests import get
from urllib import urlretrieve
import os, json, cv2

url = 'http://vision.cs.uiuc.edu/pascal-sentences/'
html = soup(get(url).text)
tds = html.find_all('td')
data = {}
cn = 0
shp = (500, 300)

for td in tds:
    imgs = td.find_all('img')
    if imgs:
        cn += 1
        print cn

        img = imgs[0]
        src = '{}{}'.format(url, img['src'])
        filename = img['src'].replace('/', '_')
        path = os.path.join('./data/pascal1k/', filename)
        urlretrieve(src, path)

        img = cv2.imread(path)
        rs = cv2.resize(img, shp)
        cv2.imwrite(path, rs)
    texts = td.find_all('td')
    if texts:
        data[filename] = [ t.text[1:] for t in texts ]

with open('./data/pascal1k/data.json', 'w') as outfile:
    json.dump(data, outfile)
