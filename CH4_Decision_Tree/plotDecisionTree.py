# coding: utf-8
import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def getNumLeafs(node)->int:
    """
      getNumLeafs：计算树的叶子节点数量
      :param node:根结点
      :returns  numLeafs:根节点数量
      """
    numLeafs=0
    if node.isLeaf:
        numLeafs += 1
    else:
        for child in node.children:#依次递归调用
            numLeafs += getNumLeafs(child)
    return numLeafs

def getTreeDepth(node):
    """
    getTreeDepth：计算树的最大深度
    :param node:根结点
    :returns  maxDepth:树的最大深度
    """
    maxDepth=0
    for child in node.children:
        if child:
            thisDepth=1+getTreeDepth(child)
        else: thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    """
    plotNode：画节点
    """
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',va="center", ha="center",\
    bbox=nodeType,arrowprops=arrow_args,fontsize = 12)

#画箭头上的文字
def plotMidText(cntrPt,parentPt,txtString):
    # lens=len(txtString)
    # xMid=(parentPt[0]+cntrPt[0])/2.0-lens*0.002
    # yMid=(parentPt[1]+cntrPt[1])/2.0
    #createPlot.ax1.text(xMid, yMid, txtString)
    xMid = 2 * (parentPt[0] - cntrPt[0]) / 5.0 + cntrPt[0]
    yMid = 2 * (parentPt[1] - cntrPt[1]) / 5.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, fontsize=12)


def plotTree(node,parentPt,nodeTxt):
    numLeafs=getNumLeafs(node)
    depth=getTreeDepth(node)
    firstStr=node.attr
    cntrPt=(plotTree.x0ff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    children=node.children
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD
    for child in children:
        if child.children:
            plotTree(child,cntrPt,child.value)
        else:
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW
            plotNode(child.label,(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,child.value)
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD

def createPlot(decision_tree):
    node = decision_tree.root
    fig=plt.figure(1,facecolor='white',figsize=(10, 8))
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(node))
    plotTree.totalD=float(getTreeDepth(node))
    plotTree.x0ff=-0.5/plotTree.totalW
    plotTree.y0ff=1.0
    plotTree(node,(0.5,1),'')
    plt.show()



