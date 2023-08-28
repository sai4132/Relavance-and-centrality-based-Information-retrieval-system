#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import modules
import numpy as np
import pandas as pd
import re
import networkx as nx
import os


# In[2]:


import csv
import xml.etree.ElementTree as ET

# Opening CSV file which will store page ids, titles and contents.
with open('pages.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # Headers for the CSV file
    column_names = ['page_id', 'title', 'content']
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()

    # Open XML file
    tree = ET.iterparse('enwiki-20230220-pages-articles-multistream2.xml', events=('start', 'end'))

    # Iterate in the XML file to get page information
    for event, page in tree:
        if event == 'end' and page.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
            # Extracting page ID, title, and content of the page
            page_id = page.find('{http://www.mediawiki.org/xml/export-0.10/}id').text
            title = page.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
            content = page.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text

            # write in CSV file
            writer.writerow({'page_id': page_id, 'title': title, 'content': content})
            
            page.clear()


# In[3]:


with open('enwiki-20230220-pages-articles-multistream-index2.txt', 'r') as infile, open('page_index.txt', 'w') as outfile:
    for line in infile:
        text = line[line.index(':')+1:]
        outfile.write(text)


# In[4]:


titles=[] # to store all titles
ids=[] # to store all page ids
with open('page_index.txt', 'r') as file:
    for line in file:
        ids.append(line[:line.index(':')])
        titles.append(line[line.index(':')+1:-1].lower())


# In[5]:


titles=np.array(titles)
ids=np.array(ids)
indices=np.argsort(titles)
titles=titles[indices] # Sorting titles to perform fast searching
ids=ids[indices] # rearranging page ids according to sorted titles


# In[6]:


def find_id(title,titles):
    '''
    Find the page id of the given title
    '''
    low = 0
    high = len(titles) - 1

    while low <= high:
        mid = (low + high) // 2
        if titles[mid] == title:
            return ids[mid]
        elif titles[mid] < title:
            low = mid + 1
        else:
            high = mid - 1

    return -1 # If page id not found


# In[7]:


pages=pd.read_csv('pages.csv') # read page ids, titles and content of the pages


# In[8]:


pages.dropna(inplace=True) # Drop pages which have no titles


# In[9]:


def outlink(content):
    '''
    Enter the page content as an input and the function will
    find all outlinks from the content. The outlinks are generally in 
    enclosed within the brackets [[ and ]].
    With in [[ ,]] sometimes image files are also there and we will ignore those.
    
    Many of the outlinks have anchor text and they are written in the form [[outlink title | anchor text]] ,
    we will only keep the outlink titles.
    '''
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, content)
    for i in range(len(matches)):
        try:
            # Ignore image files 
            if 'file' in matches[i].lower():
                matches[i]=matches[i][matches[i].index('[[')+2:]
        except:
            pass
        try:
            # Ignore the anchor text
            if '|' in matches[i]:
                j=matches[i].index('|')
                matches[i]=matches[i][:j]
        except:
            pass
    outlink_ids=[]
    for match in matches:
        id_=find_id(match.lower(),titles)
        if id_!=-1:
            outlink_ids.append(id_)
    return outlink_ids


# In[10]:


pages['oulink']=pages['content'].apply(outlink) # Find outlink of all the pages


# In[11]:


outlink_df=pages.copy() # A copy of pages dataframe which will contain only the page id, and page ids of outlinks


# In[12]:


outlink_df = outlink_df.drop(['title', 'content'], axis=1)


# In[13]:


outlink_df.to_csv("outlink.csv",index=False) # Writing the outlink_df to a csv file


# In[14]:


nodes=list(outlink_df['page_id']) # ALl the page ids which will be nodes to a graph object from networkx module


# In[15]:


edge_list=[] # Will contain all the directed edges for the graph
for i in range(len(nodes)):
    outlinks=outlink_df.iloc[i,1]
    for j in range(len(outlinks)):
        edge_list.append((nodes[i],int(outlinks[j])))


# In[16]:


G = nx.Graph()  # Create a graph  object


# In[17]:


G.add_nodes_from(nodes) # Add nodes to the graph


# In[18]:


G.add_edges_from(edge_list) # Add edges to the graph


# In[19]:


# Pagerank
pageRank = nx.pagerank(G)


# In[20]:


# Hub and Authority scores
hub, authority = nx.hits(G)


# In[21]:


# TOP 5 Pages by PageRank
sorted_PR = sorted(pageRank.items(), key=lambda x: x[1], reverse=True)
top_5_PR_ids = [k for k, v in sorted_PR[:5]]
print("Pages with top 5 PageRank : ")
for i in range(len(top_5_PR_ids)):
    index=np.where(ids==str(top_5_PR_ids[i]))[0][0]
    print(f"{i+1}. Page_id : {top_5_PR_ids[i]}, Title : {titles[index]}, PageRank : {pageRank[top_5_PR_ids[i]]}")
del sorted_PR
del top_5_PR_ids


# In[22]:


# TOP 5 Pages by Hub
sorted_hub = sorted(hub.items(), key=lambda x: x[1], reverse=True)
top_5_hub_ids = [k for k, v in sorted_hub[:5]]
print("Pages with top 5 hub score : ")
for i in range(len(top_5_hub_ids)):
    index=np.where(ids==str(top_5_hub_ids[i]))[0][0]
    print(f"{i+1}. Page_id : {top_5_hub_ids[i]}, Title : {titles[index]}, Hub score : {hub[top_5_hub_ids[i]]}")
del sorted_hub
del top_5_hub_ids


# In[23]:


# TOP 5 Pages by Authority
sorted_authority = sorted(authority.items(), key=lambda x: x[1], reverse=True)
top_5_authority_ids = [k for k, v in sorted_authority[:5]]
print("Pages with top 5 Authority score : ")
for i in range(len(top_5_authority_ids)):
    index=np.where(ids==str(top_5_authority_ids[i]))[0][0]
    print(f"{i+1}. Page_id : {top_5_authority_ids[i]}, Title : {titles[index]},Authority score : {authority[top_5_authority_ids[i]]}")
del sorted_authority
del top_5_authority_ids


# In[ ]:





# In[24]:


# whoosh


# In[25]:


os.makedirs('index', exist_ok=True)


# In[24]:


from whoosh.fields import Schema, TEXT, ID,KEYWORD
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh.qparser import MultifieldParser
from whoosh.index import open_dir
from whoosh.scoring import TF_IDF
from whoosh.scoring import BM25F


# In[26]:


schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True),page_id=ID(stored=True))
ix = create_in("index", schema)
writer = ix.writer()
for i in range(len(pages)): 
    writer.add_document(title=pages.iloc[i,1], content=pages.iloc[i,2],page_id = str(pages.iloc[i,0]))


# In[27]:


writer.commit()


# In[29]:


from whoosh import index
ix = index.open_dir("index")


# In[47]:


def bm25_scores(query):
    '''
    Find BM25 scores/weights of relevant pages for a given a query
    '''
    score_dic={}
    with ix.searcher(weighting=BM25F()) as searcher_bm25:
        parser = MultifieldParser(["title","content"],ix.schema)
        query = parser.parse(query)
        results = searcher_bm25.search(query,limit=None)
        for i in range(len(results)):
            hit = results[i]
            fields = hit.fields()
            content = fields['content']
            title = fields['title']
            page_id=fields['page_id']
            score = hit.score
            score_dic[(page_id,title)]=score
        # print("Hit #{}:".format(i+1))
        # print("==========")
        # print("TITLE :",title)
        # #print("CONTENTS :",content)  # Uncomment this if you want to see the content of the page
        # print("SCORE :", score)
    return score_dic


# In[48]:


def tfidf_scores(query):
    '''
    Find TFIDF scores/weights of relevant pages for a given a query
    '''
    score_dic={}
    with ix.searcher(weighting=TF_IDF()) as searcher_tfidf:
        parser = MultifieldParser(["title","content"],ix.schema)
        query = parser.parse(query)
        results = searcher_tfidf.search(query,limit=None)
        for i in range(len(results)):
            hit = results[i]
            fields = hit.fields()
            content = fields['content']
            title = fields['title']
            page_id=fields['page_id']
            score = hit.score
            score_dic[(page_id,title)]=score
        # print("Hit #{}:".format(i+1))
        # print("==========")
        # print("TITLE :",title)
        # #print("CONTENTS :",content)  # Uncomment this if you want to see the content of the page
        # print("SCORE :", score)
    return score_dic


# In[67]:


def weighted_scores(query,wt,wb,wp,wh):
    '''
    wt: weight for tfidf
    wb: weight for bm25
    wp: weight for pageRank
    wh: weight for hitts
    '''
    # Normalize TFIDF and BM25 values so they are in the interval [0,1]
    tfidfScores=tfidf_scores(query)
    bm25Scores=bm25_scores(query)
    max_tfidf=max(tfidfScores.values())
    max_bm25=max(bm25Scores.values())
    for key in tfidfScores.keys():
        tfidfScores[key]=tfidfScores[key]/max_tfidf
    for key in bm25Scores.keys():
        bm25Scores[key]=bm25Scores[key]/max_bm25
    result_ids=list(set(tfidfScores.keys()).union(bm25Scores.keys()))
    
    weighted_score={}
    
    # Find maximum hub,PR and authority values for scaling purposes.
    max_pr=max(list(pageRank.values()))
    max_hub=max(list(hub.values()))
    max_auth=max(list(authority.values()))
    # Calculate weights
    for pid in result_ids:
        weighted_score[pid]=wt*tfidfScores[pid]+wb*bm25Scores[pid]
        # We add scaled PR, hub and authority values such that maximum value of
        # of PR, hub and authority of a page is 1.
        weighted_score[pid]+=wp*pageRank[int(pid[0])]/max_pr # Adding a scaled PR
        # Adding Scaled Hub and authorities
        weighted_score[pid]+=(wh/2)*(hub[int(pid[0])]/max_hub+authority[int(pid[0])]/max_auth)
    weighted_score=sorted(weighted_score.items(), key=lambda x: x[1], reverse=True)
    return weighted_score


# # Title search and scoring

# In[30]:


np.random.seed(1)
random_titles= np.random.choice(titles, 5, replace=False) # Picking random titles


# In[53]:


for title in random_titles:
    print('Title is :',title)
    print("BM25 scores for the title are : ")
    bm25_score_=sorted(bm25_scores(title).items(), key=lambda x: x[1], reverse=True)[:1]
    print(bm25_score_)
    print("\n")
    print("TFIDF scores for the title are : ")
    tfidf_score_=sorted(tfidf_scores(title).items(), key=lambda x: x[1], reverse=True)[:1]
    print(tfidf_score_)
    print("\n")


# In[69]:


for title in random_titles:
    print('Title is :',title)
    print('Weighted Scores(Combining relevance and centrality measures) for the title are:')
    print(weighted_scores(title,0.25,0.25,0.25,0.25)[:1])
    print("\n")


# In[70]:


for title in random_titles:
    print('Title is :',title)
    print('Weighted Scores(Combining relevance and centrality measures) for the title are:')
    print(weighted_scores(title,0.20,0.40,0.20,0.20)[:1])
    print("\n")


# In[71]:


for title in random_titles:
    print('Title is :',title)
    print('Weighted Scores(Combining relevance and centrality measures) for the title are:')
    print(weighted_scores(title,0.05,0.35,0.30,0.30)[:1])
    print("\n")


# In[ ]:




