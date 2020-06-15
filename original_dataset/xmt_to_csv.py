# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:33:31 2020

@author: Kartik
"""

from xml.etree import ElementTree
import os
import csv

tree = ElementTree.parse("ABSA15_Laptops_Test.xml")

csv_data = open("test.csv", 'w', newline='', encoding='utf-8')
csvwriter = csv.writer(csv_data)

col_names = ['review_id', 'sentence_id', 'sentence', 'aspect_category', 'aspect_polarity']
csvwriter.writerow(col_names)

root = tree.getroot()

for review in root.findall("Review"):
    rid = review.items()[0][1]
    sentences = review.find('sentences')
    for sentence in sentences.findall('sentence'):
        row = []
        row.append(rid)
        sid = sentence.items()[0][1]
        row.append(sid)
        text = sentence.find('text').text
        row.append(text)
        opinions = sentence.find('Opinions')
        #print("opinions:", opinions)
        if opinions is not None:
            opinion = opinions.findall('Opinion')
            l = []
            for item in opinion:
                op = {}
                op['category'] = item.items()[0][1]
                op['polarity'] = item.items()[1][1]
                l.append(op)
            row.append(l)
        else:
            row.append('')
        
        csvwriter.writerow(row)

print("Done")
                