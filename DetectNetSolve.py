import shutil
import os
curPath = os.path.dirname(os.path.abspath(__file__))
# MUST change current directory to actual caffe path!!!
caffeRoot = '/home/gnoses/Project/DetectNet/caffe-caffe-0.15/'
# os.chdir(caffeRoot)
import sys
sys.path.insert(0, caffeRoot + 'python')
sys.path.insert(0, '/usr/local/cuda/lib64/')

import caffe
from TrainingPlot import *

def MakeResultData(plt, solverFilename, resultText):
    savePath = 'examples/Cityscape/result/' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    os.makedirs(savePath)
    plt.savefig(savePath + '/result_figure.png')
    shutil.copy(solverFilename, savePath + '/result_solver.prototxt')

    fp = open(savePath + '/result.txt', 'wt')
    for text in resultText:
        print >>fp, text
    fp.close()



def Train(solverFilename, caffeModel, useGPU, trainIter, testIter):

    if (useGPU):
        caffe.set_device(0)
        caffe.set_mode_gpu()

    print('Start')

    solver = caffe.SGDSolver(solverFilename)


    #caffeModel = 'examples/Cityscape/snapshot_iter_25000.caffemodel'
    if (caffeModel != None):
        solver.net.copy_from(caffeModel)


    # For reference, we also create a solver that does no finetuning.

    trainingPlot = TrainingPlot()
    trainingPlot.SetConfig(5, 1000, trainIter)
    for it in range(trainIter):
        solver.step(1)  # SGD by Caffe
        # print solver.net.blobs
        trainLoss = solver.net.blobs['loss_bbox'].data.item(0) + solver.net.blobs['loss_coverage'].data.item(0)
        trainAcc = 0 # solver.net.blobs['mAP'].data.item(0)


        if (it % 2 == 0):
            #watch = StopWatch()
            solver.test_nets[0].forward()
            valLoss = solver.test_nets[0].blobs['loss_bbox'].data.item(0) + solver.test_nets[0].blobs['loss_coverage'].data.item(0)
            valAcc = solver.test_nets[0].blobs['mAP'].data.item(0)
            #watch.PrintCheckTime('test time')

        trainingPlot.Add(it, trainLoss, valLoss, trainAcc, valAcc)
        trainingPlot.Show()

    print 'Training finished...'


    testAcc = 0
    scratch_accuracy = 0
    # print 'Start testing'
    #
    # for it in range(testIter):
    #     solver.test_nets[0].forward()
    #     acc = solver.test_nets[0].blobs['accuracy'].data
    #     testAcc += acc
    #     if (it % 100 == 0):
    #         text = 'Test #%d / %d : %f' % (it, testIter, acc)
    #         print text
    #         resultText.append(text)
    #
    # testAcc /= testIter
    #scratch_accuracy /= test_iters
    # print 'Test Accuracy for fine-tuning:', testAcc
    #MakeResultData(plt, solverFilename, resultText)


# main
mode = 0

caffeModel = None

# basic small 1000 dataset
if (mode == 0):
    solverFilename = curPath + '/detectnet_solver.prototxt'
    # netFilename = curPath + '/Models/bayesian_segnet_train10K_64feature/bayesian_segnet_basic_train.prototxt'
    # solverFilename = '/home/gnoses/SegNet/Models/bayesian_segnet_basic_solver.prototxt'
    caffeModel = curPath + '/bvlc_googlenet.caffemodel'


#useGPU = True
useGPU = False
#caffeModel = curPath + '/snapshotCamvid/bayesian_segnet_basic_iter_1050.caffemodel'
# params : solverFilename, trainIter, testIter
Train(solverFilename, caffeModel, useGPU, 10000, 1000)
