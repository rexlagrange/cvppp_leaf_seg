#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
import h5py
import cv2



def evaluate(inLabelFileName, gtLabelFileName):

    inLabelFile = h5py.File(inLabelFileName,'r')
    gtLabelFile = h5py.File(gtLabelFileName,'r')
    
    # loop all datasets in gtLabelFile and compare to respective dataset in inLabelFile
    results = {'header' : ('number', 'filename', 'diffFG', 'bestDice', 'FgBgDice')}
    stats = {'header' : ('measure','mean', 'std')}
    for key in gtLabelFile.keys(): # loop A1 to A5
        group = gtLabelFile.get(key)
        diffFG = []
        bestDice = []
        FgBgDice = []
        filenames = []
        imgNum = 0
        for data in group.keys(): # loop datasets in the current group
            # get label image
            fullkey = key + '/' + data + '/label' 
            
            print(fullkey)
            
            gtLabel = np.asarray(gtLabelFile.get(fullkey))
            if fullkey in inLabelFile:
                inLabel = np.asarray(inLabelFile.get(fullkey))
            else:
                inLabel = np.zeros(gtLabel.shape)

            # now call the scoring routines and append results to respective lists                
            diffFG.append( DiffFGLabels(inLabel,gtLabel))
            bestDice.append( BestDice(inLabel,gtLabel))
            FgBgDice.append( FGBGDice(inLabel,gtLabel))
            
            imgNum += 1
            
            #  get the original filename 
            fullkey = key + '/' + data + '/label_filename'
            filename = str(np.asarray(gtLabelFile.get(fullkey)))
            filenames.append(filename)
            
        # store results in dictionary
        zipped = zip(range(1,imgNum+1),filenames,diffFG,bestDice,FgBgDice)   
        groupresults = {key : list(zipped)}
        groupstats = {key : [('diffFG', np.mean(diffFG), np.std(diffFG)),
                             ('absDiffFG', np.mean(np.abs(diffFG)), np.std(np.abs(diffFG))),
                             ('bestDice', np.mean(bestDice), np.std(bestDice)),
                             ('FgBgDice', np.mean(FgBgDice), np.std(FgBgDice))]}
        results.update(groupresults)
        stats.update(groupstats)
        
    gtLabelFile.close()
    inLabelFile.close()
    
    return results, stats

def evaluate_without_h5(inLabeldir, gtLabeldir):
    # loop all datasets in gtLabelFile and compare to respective dataset in inLabelFile
    results = {'header' : ('number', 'filename', 'diffFG', 'bestDice', 'FgBgDice')}
    stats = {'header' : ('measure','mean', 'std')}
    for subdir in os.listdir(inLabeldir): # loop A1 to A5
        diffFG = []
        bestDice = []
        FgBgDice = []
        filenames = []
        imgNum = 0
        for data in os.listdir(os.path.join(inLabeldir,subdir)): # loop datasets in the current group
            # get label image
            fullkey = os.path.join(inLabeldir, subdir,data) 
            print("comparing",fullkey,end=" with ")
            gtlablekey=os.path.join(gtLabeldir, subdir,data)
            print(gtlablekey)
            gtLabel = np.asarray(cv2.imread(gtlablekey, cv2.IMREAD_UNCHANGED))
            inLabel = np.asarray(cv2.imread(fullkey, cv2.IMREAD_UNCHANGED))
            # now call the scoring routines and append results to respective lists                
            diffFG.append( DiffFGLabels(inLabel,gtLabel))
            bestDice.append( BestDice(inLabel,gtLabel))
            FgBgDice.append( FGBGDice(inLabel,gtLabel))
            imgNum += 1
            filenames.append(data)
            
        # store results in dictionary
        zipped = zip(range(1,imgNum+1),filenames,diffFG,bestDice,FgBgDice)   
        groupresults = {subdir : list(zipped)}
        groupstats = {subdir : [('diffFG', np.mean(diffFG), np.std(diffFG)),
                             ('absDiffFG', np.mean(np.abs(diffFG)), np.std(np.abs(diffFG))),
                             ('bestDice', np.mean(bestDice), np.std(bestDice)),
                             ('FgBgDice', np.mean(FgBgDice), np.std(FgBgDice))]}
        results.update(groupresults)
        stats.update(groupstats)
    return results, stats
def evaluate_without_h5_nosubfolder(inLabeldir, gtLabeldir):
        # loop all datasets in gtLabelFile and compare to respective dataset in inLabelFile
    results = {'header' : ('number', 'filename', 'diffFG', 'bestDice', 'FgBgDice')}
    stats = {'header' : ('measure','mean', 'std')}
    diffFG = []
    bestDice = []
    FgBgDice = []
    filenames = []
    imgNum = 0
    for data in os.listdir(inLabeldir): # loop datasets in the current group
        # get label image
        fullkey = os.path.join(inLabeldir, data) 
        print("comparing",fullkey,end=" with ")
        gtlablekey=os.path.join(gtLabeldir, data)
        print(gtlablekey)
        gtLabel = np.asarray(cv2.imread(gtlablekey, cv2.IMREAD_UNCHANGED))
        inLabel = np.asarray(cv2.imread(fullkey, cv2.IMREAD_UNCHANGED))
        # now call the scoring routines and append results to respective lists                
        diffFG.append( DiffFGLabels(inLabel,gtLabel))
        bestDice.append( BestDice(inLabel,gtLabel))
        FgBgDice.append( FGBGDice(inLabel,gtLabel))
        imgNum += 1
        filenames.append(data)
        
    # store results in dictionary
    zipped = zip(range(1,imgNum+1),filenames,diffFG,bestDice,FgBgDice) 
    curdir=os.path.basename(inLabeldir)   
    groupresults = {curdir : list(zipped)}
    groupstats = {curdir : [('diffFG', np.mean(diffFG), np.std(diffFG)),
                            ('absDiffFG', np.mean(np.abs(diffFG)), np.std(np.abs(diffFG))),
                            ('bestDice', np.mean(bestDice), np.std(bestDice)),
                            ('FgBgDice', np.mean(FgBgDice), np.std(FgBgDice))]}
    results.update(groupresults)
    stats.update(groupstats)
    return results, stats
##############################################################################
def DiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    maxInLabel = int(np.max(inLabel)) # maximum label value in inLabel
    minInLabel = int(np.min(inLabel)) # minimum label value in inLabel
    maxGtLabel = int(np.max(gtLabel)) # maximum label value in gtLabel
    minGtLabel = int(np.min(gtLabel)) # minimum label value in gtLabel

    return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel) 


##############################################################################
def BestDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: score: Dice score
#
# We assume that the lowest label in inLabel is background, same for gtLabel
# and do not use it. This is necessary to avoid that the trivial solution, 
# i.e. finding only background, gives excellent results.
#
# For the original Dice score, labels corresponding to each other need to
# be known in advance. Here we simply take the best matching label from 
# gtLabel in each comparison. We do not make sure that a label from gtLabel
# is used only once. Better measures may exist. Please enlighten me if I do
# something stupid here...

    score = 0 # initialize output
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return score
    
    maxInLabel = np.max(inLabel) # maximum label value in inLabel
    minInLabel = np.min(inLabel) # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel
    
    if(maxInLabel==minInLabel): # trivial solution
        return score
    
    for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j) # compare labelled regions
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    score = score/(maxInLabel-minInLabel)
    return score

##############################################################################
def FGBGDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
#        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
# output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel) # minimum label value in inLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    one = np.ones(inLabel.shape)    
    inFgLabel = (inLabel != minInLabel*one)*one
    gtFgLabel = (gtLabel != minGtLabel*one)*one
    
    return Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground

    
##############################################################################
def Dice(inLabel, gtLabel, i, j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0

    return out    
    
##############################################################################
def AbsDiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: Absolute value of difference of the number of foreground labels

    return np.abs( DiffFGLabels(inLabel,gtLabel) )
    
##############################################################################
def WriteOutput(output_filename,somedict):
# output_filename: name of the output file
# results: array containing the result values
    output_file = open(output_filename, 'w')
    # write header if available
    if 'header' in somedict:
        output_file.write('dataset,')
        output_file.write(','.join(map(str,somedict['header'])))
        output_file.write('\n')
    # write rest
    for key in somedict:
        # skip header
        if key.find('header')==-1:
            # get every line and output it 
            for line in somedict[key]:
                output_file.write(key)
                output_file.write(',')
                output_file.write(','.join(map(str,line)))
                output_file.write('\n')
        
    output_file.close()

##############################################################################
# main routine
if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    
    if not os.path.isdir(submit_dir):
        print(submit_dir + " doesn't exist")
    
    # check if input directories exist
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        # create output dir if not already there
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # the input filenames
        truth_file = os.path.join(truth_dir, "CVPPP2017_testing_truth.h5")
        submission_answer_file = os.path.join(submit_dir, "answer.h5")

        # call the evaluation routine dealing with these files
        results, stats = evaluate(submission_answer_file, truth_file)

        # write results to output file
        output_filename = os.path.join(output_dir, 'details.txt')    
        WriteOutput(output_filename,results)
        output_filename = os.path.join(output_dir, 'scores.txt')    
        WriteOutput(output_filename,stats)
        






























