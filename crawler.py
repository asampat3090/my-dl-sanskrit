import urllib
import urllib2
import os
import itertools
from bs4 import BeautifulSoup

print "\n\nSTART"

newpages = set()

top_dir = "http://gretil.sub.uni-goettingen.de/gret_utf.htm"
res = urllib2.urlopen(top_dir)
html_code = res.read()
bs_obj = BeautifulSoup(html_code)
# Get file links 

file_links = []
suffix = []
try :
	print "trying to get all links"
	for a in bs_obj.findAll('a'):
		if a.get('href')!="" and a.get('href') is not None:
			file_links.append('http://gretil.sub.uni-goettingen.de/'+a.get('href').split('http://gretil.sub.uni-goettingen.de/')[0])
except Exception: # Magnificent exception handling
    pass



# Loop through file links
base_path = '/Users/anandsampat/dl-sanskrit/data/'
for link in file_links:
	if link!='http://gretil.sub.uni-goettingen.de/':
		link_path=link.split('http://gretil.sub.uni-goettingen.de/')
		print link_path
		if len(link_path)>1:
			link_arr=link_path[1].split('/')
			# Loop through link_arr to create or verify file structure
			# only if the final file ends in .htm though
			if (len(link_arr)>1 or link_arr[0]!='') and link_arr[-1][-4:]==".htm":
				sub_path = ''
				for dir_name in link_arr[:-1]:
					print dir_name
					if dir_name!="":
						sub_path = sub_path + dir_name + '/'
						if not os.path.isdir(''.join((base_path,sub_path))):	
							# create sub directory if not already present
							print "Making directory...." + base_path + sub_path
							os.mkdir(base_path+sub_path)
				# write to file in the given path
				print "Downloading from...." + link
				print "Downloading to...." + base_path + sub_path + link_arr[-1]
				urllib.urlretrieve(link,base_path+sub_path+link_arr[-1]) 


#for k,v in artist_pages: 
#	try: 
#		c = urllib2.urlopen(v)
#	except: 
#		print "Could not open %s" % v
#		continue
#	soup = BeautifulSoup(c.read())
#	links = soup('a')
#	for link in links: 	
#		if ('href' in dict(link.attrs)):
#			url = link['href']
#			# add to list only if the last 5 chars are .html and it not top100 link
#			taboo_list=["http://www.metrolyrics.com/videos.html","http://www.metrolyrics.com/news.html","http://www.metrolyrics.com/top-artists.html","http://www.metrolyrics.com/rolling-stone-top500.html","http://metrolyrics.com/add.html"]
#			if url[-5:] == ".html" and url[:33]!="http://www.metrolyrics.com/top100" and url not in taboo_list:
#				# add to the list 
#				url_list.add(url)
#				names_list.append(k)
#print len(url_list)
