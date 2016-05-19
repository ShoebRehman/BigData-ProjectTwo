# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:36:38 2016

@author: Shoeb
"""

from __future__ import print_function

import sys
from operator import add
import math
from pandas import DataFrame
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tfidf <file> queryTerm", file=sys.stderr)
        exit(-1)
    try:
        
        sc = SparkContext(appName="CosSim")
        lines = sc.textFile(sys.argv[1], 1) #takes first parameter and creates an RDD based on it
        lineCount = lines.filter(lambda line: "gene_" in line or "disease_" in line).count() #returns count of lines with gene or disease
        queryTerm = sys.argv[2] #stores queryTerm into variable
        
		#intakes format of docid t1...tn and outputs a format of ((T,docid),tfidf))
        tfidfRDD = lines.filter(lambda line: "gene_" in line or "disease_" in line).map(lambda x: (x.split(' ')[0],x.split(' ')[1:]))\
                      .flatMap(lambda (k,v): [(k, x) for x in v])\
                      .map(lambda x: ((x[1], x[0]),float(1))).filter(lambda k: "gene_" in k[0][0] or "disease_" in k[0][0])\
                      .reduceByKey(add)\
                      .map(lambda x: (x[0][1],(x[0][0],x[1])))\
                      .groupByKey().mapValues(list)\
                      .map(lambda x: ((x[0],sum(int(v) for name,v in x[1])),x[1]))\
                      .flatMapValues(lambda x: x)\
                      .map(lambda x: ((x[1][0],x[0][0]),float(x[1][1]/x[0][1])))\
                      .map(lambda b: (b[0][0],(b[0][1],b[1]))).groupByKey().mapValues(list)\
                      .flatMap(lambda (k,v): [((k,x),len(v)) for x in v])\
                      .map(lambda x: ((x[0][0],x[0][1][0]), x[0][1][1]*math.log(float(lineCount/x[1]),math.e)))\

        tfidfMap = tfidfRDD.collectAsMap() #collects RDD as a map. format for this is ((term,docid),tfidf)
        
        '''
        newOutput = dict()
        for value in tfidfMap.keys():
            if(not value[0] in newOutput.keys()):
                newOutput[value[0]] = dict()       
            newOutput[value[0]][value[1]] = tfidfMap[value]# reorganizes the dictionary from (K1,K2):V to [K1][K2]:V for dataframe processing
        '''
                    
        tfidfArray = tfidfRDD.map(lambda x: (x[0][0],(x[0][1],x[1])))\
                           .groupByKey().mapValues(list) #gives us (term,{(D,tfidf)})     
                           
        tfidfArrayOutput = tfidfArray.collect() #collects as list
        with open('tfidfs.txt','w') as f:#stores the tfidfs into output file for testing purposes                
            for row in sorted(tfidfArrayOutput,key=lambda x:x[0]):
                f.write("%s : %s" % (row[0],row[1]))
                f.write("\n")
                
        queryDocList = tfidfArray.filter(lambda x: queryTerm in x[0]).flatMapValues(lambda x: x).map(lambda x: (x[1][0],x[1][1])).collectAsMap() #collects queryTerm's doc and tfidf values as map
        
        norm = tfidfArray.flatMapValues(lambda x: x).map(lambda x: (x[0],math.pow(x[1][1],2))).groupByKey().map(lambda x: (x[0],math.sqrt(sum(x[1])))).collectAsMap() #returns norms of all terms
        
        with open('norms.txt','w') as f:#rewrites all norms into text file for testing purposes
            for row in sorted(norm.keys(),key=lambda x:x[0]):
                f.write("%s : %s" % (row,norm[row]))
                f.write("\n")
                
        
        tfidfOutput = tfidfArray.flatMapValues(lambda x: x)
                
        commonDocTFIDF = tfidfOutput.filter(lambda x: x[1][0] in queryDocList.keys())\
                                         .map(lambda y: y).groupByKey().mapValues(list)\
                                         .map(lambda x: (x[0],sorted(x[1],key=lambda y: float(y[0][3:]))))\
                                         .flatMapValues(lambda x: x) #filters out all of the terms that don't share a common doc with queryTerm
        
        num = commonDocTFIDF.filter(lambda x:"disease_" in x[0] and "_disease" in x[0] and not "_disease_" in x[0])\
                            .map(lambda x: (x[0],queryDocList[x[1][0]]*x[1][1])).reduceByKey(add)\
                            .map(lambda x: (x[0],x[1]/(norm[x[0]]*norm[queryTerm]))).collect()#finds the numerator of the query term and every other term, multiples tfidfs and gets the sum of products
							#multiplies vector magnitudes and divides the numerator with vector magnitudes calculated above (norm)
                            
        with open('queryTermSimOutput.txt','w') as f:                
            for row in sorted(num,key=lambda x:x[1],reverse=True): #writes sorted in desc order of all terms related to queryTerm and by how much
                f.write("%s : %s" % (row[0],row[1]))
                f.write("\n")
        '''        
        #stores tfidfs into dataframe and writes to file    
        df = DataFrame(newOutput).T.fillna(0)
        df.reindex_axis(sorted(df.columns, key=lambda x: float(x[3:])), axis=1)
        df.to_csv('output.csv', sep=',')
        '''
        
        sc.stop()

    except ValueError:
        print (ValueError)
        sc.stop()