import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from SVC_Utils import *
from sklearn.metrics import roc_curve, auc


# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_target_net(net, testloader, classes=None):

    if classes is not None:
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
    total = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (imgs, lbls) in enumerate(testloader):

            imgs, lbls = imgs.to(device), lbls.to(device)

            output = net(imgs)

            predicted = output.argmax(dim=1)

            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()

            if classes is not None:
                for prediction, lbl in zip(predicted, lbls):

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1
                    
    accuracy = 100*(correct/total)
    if classes is not None:
        for i in range(len(classes)):
            print('Accuracy of %s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("\nAccuracy = %.2f %%\n\n" % (accuracy) )
    
    return accuracy



def eval_target_net_entropic(model1=None, model2=None, data_loader=None, classes=None, alpha=1.001, class_number=None):
    """
    Function to evaluate a target model provided
    specified data sets.

    Parameters
    ----------
    model       : Module
                  PyTorch conforming nn.Module function
    data_loader : DataLoader
                  PyTorch dataloader function
    classes     : list
                  list of classes
    alpha       : float
                  parameter of Renyi Entropy
    Returns
    -------
    accuracy    : float
                  accuracy of target model
    """

    if classes is not None:
        n_classes = len(classes)
        class_correct1 = np.zeros(n_classes)
        class_total1 = np.zeros(n_classes)
        average_entropy_per_class1 = np.zeros(n_classes)
        class_correct2 = np.zeros(n_classes)
        class_total2 = np.zeros(n_classes)
        average_entropy_per_class2 = np.zeros(n_classes)
    class_label = 0

    total1 = 0
    correct1 = 0
    total_entropy1 = 0
    total_entropy_correct1 = 0
    total_entropy_wrong1 = 0
    total2 = 0
    correct2 = 0
    total_entropy2 = 0
    total_entropy_correct2 = 0
    total_entropy_wrong2 = 0

    total_entropy_correct_squared1 = 0
    total_entropy_wrong_squared1 = 0
    total_entropy_squared1 = 0
    total_entropy_correct_squared2 = 0
    total_entropy_wrong_squared2 = 0
    total_entropy_squared2 = 0

    with torch.no_grad():
        model1.eval()
        model2.eval()
        for i, (imgs, lbls) in enumerate(data_loader):

            
            # redact images belonging to class number
            # introduce new label as zero
            lbls1 = (lbls-1) * (lbls.gt(class_number)).long() + (lbls) * (1 - lbls.ge(class_number)).long()

            # mask out images in class_number
            imgs1 = imgs * torch.ones( imgs.size() )
            imgs1 = torch.mm(torch.diag((1.0-lbls1.eq(0))).float(), imgs1.view(imgs.size()[0],-1))
            imgs1 = imgs1.view(imgs.size())
            imgs1, lbls1 = imgs1.to(device), lbls1.to(device)
            
            

            #shift do to argmax
            lbls2 = (lbls-1) * (lbls.gt(class_number)).long() + (lbls) * (1 - lbls.ge(class_number)).long() 
                   
            imgs2, lbls2 = imgs.to(device), lbls2.to(device)
            


            output1 = model1(imgs1)
          
            output2 = model2(imgs2)

            # truncate redaction
            a = [j for j in range(n_classes)]
            a.remove(class_number)
            
            output1 = output1[:,a]
            output2 = output2[:,a]
            

            # print the renyi entropy of the probability distribution
            # H_a = 1/(1-a) log_i ( sum p_i^a)
            # convert to probabilities:  

            # normalize out put to probabilities 

            normalized_output1 = torch.softmax(output1, dim=1)
            normalized_output2 = torch.softmax(output2, dim=1)
           
            
            #renyi_entropy = (normalized_output  * normalized_output.log()).sum(dim=1)
            renyi_entropy1 = 1/(1-alpha) * ((torch.pow(normalized_output1, alpha)).sum(dim=1)).log()
            renyi_entropy2 = 1/(1-alpha) * ((torch.pow(normalized_output2, alpha)).sum(dim=1)).log()
            
            renyi_entropy_squared1 = renyi_entropy1 * renyi_entropy1
            renyi_entropy_squared2 = renyi_entropy2 * renyi_entropy2


            predicted1 = output1.argmax(dim=1)
            total1 += imgs.size(0)
            correct1 += predicted1.eq(lbls1).sum().item()
            total_entropy_correct1 += ((predicted1.eq(lbls1)).float() * renyi_entropy1).sum()
            total_entropy_wrong1 += ((1 - (predicted1.eq(lbls1)).float()) * renyi_entropy1).sum()
            total_entropy1 += renyi_entropy1.sum()
            total_entropy_correct_squared1 += ((predicted1.eq(lbls1)).float() * renyi_entropy_squared1).sum()
            total_entropy_wrong_squared1 += ((1 - (predicted1.eq(lbls1)).float()) * renyi_entropy_squared1).sum()
            total_entropy_squared1 += renyi_entropy_squared1.sum()


            predicted2 = output2.argmax(dim=1)
            total2 += imgs.size(0)
            correct2 += predicted2.eq(lbls2).sum().item()
            total_entropy_correct2 += ((predicted1.eq(lbls2)).float() * renyi_entropy2).sum()
            total_entropy_wrong2 += ((1 - (predicted2.eq(lbls2)).float()) * renyi_entropy2).sum()
            total_entropy2 += renyi_entropy2.sum()
            total_entropy_correct_squared2 += ((predicted2.eq(lbls2)).float() * renyi_entropy_squared2).sum()
            total_entropy_wrong_squared2 += ((1 - (predicted2.eq(lbls2)).float()) * renyi_entropy_squared2).sum()
            total_entropy_squared2 += renyi_entropy_squared2.sum()



            #print(renyi_entropy)
            #print(predicted)
            #print(lbls)

            if classes is not None:
                for entropy, prediction, lbl in zip(renyi_entropy1, predicted1, lbls1):
                    class_correct1[lbl] += (prediction == lbl).item()
                    class_total1[lbl] += 1
                    average_entropy_per_class1[lbl] += entropy.item()
                for entropy, prediction, lbl in zip(renyi_entropy2, predicted2, lbls2):
                    class_correct2[lbl] += (prediction == lbl).item()
                    class_total2[lbl] += 1
                    average_entropy_per_class2[lbl] += entropy.item()
            
      
        #average_entropy_per_class = average_entropy_per_class 
               

                    
    accuracy1 = 100*(correct1/total1)
    average_entropy_correct1 = total_entropy_correct1/correct1
    average_entropy_wrong1 = total_entropy_wrong1/(total1 - correct1)
    average_entropy1 = total_entropy1/total1
    std_entropy1 = torch.sqrt(total_entropy_squared1 / total1 - average_entropy1 * average_entropy1)
    std_entropy_correct1 = torch.sqrt(total_entropy_correct_squared1 / correct1- average_entropy_correct1 * average_entropy_correct1)
    std_entropy_wrong1 = torch.sqrt(total_entropy_wrong_squared1 / (total1 - correct1)- average_entropy_wrong1 * average_entropy_wrong1)
    if classes is not None:
        for i in range(1,len(classes)):
            if i == class_number:
                print(classes[i] + " redacted\n")
                continue
            if i < class_number:
                shift = 0
            if i > class_number:
                shift = 1 
            
            print('{} Accuracy of {}/{} : {:.3f} % average entropy {:.3f}'
                  .format(classes[i], class_correct1[i-shift] , class_total1[i-shift],
                          100 * class_correct1[i-shift] / class_total1[i-shift], 
                          average_entropy_per_class1[i-shift]/ class_total1[i-shift] ))

    print("\n Accuracy = {:.2f} %\nAverage Entropy (correct,wrong,total) = ({:.3f},{:.3f},{:.3f})\n\n".format(
         accuracy1, average_entropy_correct1, average_entropy_wrong1, average_entropy1))
    print("std (correct,wrong,total) = ({:.3f},{:.3f},{:.3f})\n".format(std_entropy_correct1, std_entropy_wrong1, std_entropy1))



    accuracy2 = 100*(correct2/total2)
    average_entropy_correct2 = total_entropy_correct2/correct2
    average_entropy_wrong2 = total_entropy_wrong2/(total2 - correct2)
    average_entropy2 = total_entropy2/total2
    std_entropy2 = torch.sqrt(total_entropy_squared2 / total2 - average_entropy2 * average_entropy2)
    std_entropy_correct2 = torch.sqrt(total_entropy_correct_squared2 / correct2 - average_entropy_correct2 * average_entropy_correct2)
    std_entropy_wrong2 = torch.sqrt(total_entropy_wrong_squared2 / (total2 - correct2) - average_entropy_wrong2 * average_entropy_wrong2)
    if classes is not None:
        for i in range(1,len(classes)):
            if i == class_number:
                print(classes[i] + " redacted\n")
                continue
            if i < class_number:
                shift = 0
            if i > class_number:
                shift = 1 
            print('{} Accuracy of {}/{} : {:.3f} % average entropy {:.3f}'
                  .format(classes[i], class_correct2[i-shift] , class_total2[i-shift],
                          100 * class_correct2[i-shift] / class_total2[i-shift], 
                          average_entropy_per_class2[i-shift]/ class_total2[i-shift] ))

    print("\n Accuracy = {:.2f} %\nAverage Entropy (correct,wrong,total) = ({:.3f},{:.3f},{:.3f})\n\n".format(
         accuracy2, average_entropy_correct2, average_entropy_wrong2, average_entropy2))
    print("std (correct,wrong,total) = ({:.3f},{:.3f},{:.3f})\n".format(std_entropy_correct2, std_entropy_wrong2, std_entropy2))

    return accuracy1

    

def eval_attack_net(attack_net, target, target_train, target_out, k):
    """Assess accuracy, precision, and recall of attack model for in training set/out of training set classification.
    Edited for use with SVCs."""
    
    in_predicts=[]
    out_predicts=[]
    losses = []
    
    if type(target) is not Pipeline:
        target_net=target
        target_net.eval()
        
    attack_net.eval()

    
    precisions = []
    recalls = []
    accuracies = []

    #for threshold in np.arange(0.5, 1, 0.005):
    thresholds = np.arange(0.5, 1, 0.005)

    total = np.zeros(len(thresholds))
    correct = np.zeros(len(thresholds))

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds))   
 
    train_top = np.empty((0,2))
    out_top = np.empty((0,2))
    
    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)
        
        #[mini_batch_size x num_classes] tensors, (0,1) probabilities for each class for each sample)
        if type(target) is Pipeline:
            traininputs=train_imgs.view(train_imgs.shape[0], -1)
            outinputs=out_imgs.view(out_imgs.shape[0], -1)
            
            train_posteriors=torch.from_numpy(target.predict_proba(traininputs)).float()
            out_posteriors=torch.from_numpy(target.predict_proba(outinputs)).float()
            
        else:
            train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
            out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)
        

        #[k x mini_batch_size] tensors, (0,1) probabilities for top k probable classes
        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:,:k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:,:k].clone().to(device)
        
        #Collects probabilities for predicted class.
        for p in train_top_k:
            in_predicts.append((p.max()).item())
        for p in out_top_k:
            out_predicts.append((p.max()).item())
        
        if type(target) is not Pipeline:
            train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))

        #print("train_top_k = ",train_top_k)
        #print("out_top_k = ",out_top_k)
        
        #print(train_top.shape)
        
        train_lbl = torch.ones(mini_batch_size).to(device)
        out_lbl = torch.zeros(mini_batch_size).to(device)
        
        #Takes in probabilities for top k most likely classes, outputs ~1 (in training set) or ~0 (out of training set)
        train_predictions = F.sigmoid(torch.squeeze(attack_net(train_top_k)))
        out_predictions = F.sigmoid(torch.squeeze(attack_net(out_top_k)))


        for j, t in enumerate(thresholds):
            true_positives[j] += (train_predictions >= t).sum().item()
            false_positives[j] += (out_predictions >= t).sum().item()
            false_negatives[j] += (train_predictions < t).sum().item()
            #print(train_top >= threshold)


            #print((train_top >= threshold).sum().item(),',',(out_top >= threshold).sum().item())

            correct[j] += (train_predictions >= t).sum().item()
            correct[j] += (out_predictions < t).sum().item()
            total[j] += train_predictions.size(0) + out_predictions.size(0)

    #print(true_positives,',',false_positives,',',false_negatives)

    for j, t in enumerate(thresholds):
        accuracy = 100 * correct[j] / total[j]
        precision = true_positives[j] / (true_positives[j] + false_positives[j]) if true_positives[j] + false_positives[j] != 0 else 0
        recall = true_positives[j] / (true_positives[j] + false_negatives[j]) if true_positives[j] + false_negatives[j] !=0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))
        

        
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    
    
def eval_attack_roc(attack_net, target_net, target_train, target_out, k):
    losses = []

    target_net.eval()
    attack_net.eval()

    total = 0
    correct = 0

    train_top = np.empty((0,2))
    out_top = np.empty((0,2))

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    predictions = np.array([])
    labels = np.array([])

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):

        train_size = train_imgs.shape[0]
        out_size = out_imgs.shape[0]
        
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)

        train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
        out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:,:k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:,:k].clone().to(device)

        train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
        out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))

        #print("train_top_k = ",train_top_k)
        #print("out_top_k = ",out_top_k)


        train_lbl = torch.ones(train_size).to(device)
        out_lbl = torch.zeros(out_size).to(device)


        train_predictions = F.sigmoid(torch.squeeze(attack_net(train_top_k)))
        out_predictions = F.sigmoid(torch.squeeze(attack_net(out_top_k)))
        
        predictions = np.concatenate((predictions, train_predictions.detach().cpu().numpy()), axis=0)
        labels = np.concatenate((labels, np.ones(train_size)), axis=0)
        predictions = np.concatenate((predictions, out_predictions.detach().cpu().numpy()), axis=0)
        labels = np.concatenate((labels, np.zeros(out_size)), axis=0)

        #print("train_predictions = ",train_predictions)
        #print("out_predictions = ",out_predictions)


        true_positives += (train_predictions >= 0.5).sum().item()
        false_positives += (out_predictions >= 0.5).sum().item()
        false_negatives += (train_predictions < 0.5).sum().item()


        correct += (train_predictions>=0.5).sum().item()
        correct += (out_predictions<0.5).sum().item()
        total += train_predictions.size(0) + out_predictions.size(0)

    accuracy = 100 * correct / total
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives !=0 else 0
    print("Membership Inference Performance")
    print("Accuracy = %.2f%%, Precision = %.2f, Recall = %.2f" % (accuracy, precision, recall))
    
    
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def eval_membership_inference(target_net, target_train, target_out):

    target_net.eval()

    precisions = []
    recalls = []
    accuracies = []

    #for threshold in np.arange(0.5, 1, 0.005):
    thresholds = np.arange(0.5, 1, 0.005)

    total = np.zeros(len(thresholds))
    correct = np.zeros(len(thresholds))

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds))

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)

        train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
        out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top = train_sort[:,0].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top = out_sort[:,0].clone().to(device)

        #print(train_top.shape)

        for j, t in enumerate(thresholds):
            true_positives[j] += (train_top >= t).sum().item()
            false_positives[j] += (out_top >= t).sum().item()
            false_negatives[j] += (train_top < t).sum().item()
            #print(train_top >= threshold)


            #print((train_top >= threshold).sum().item(),',',(out_top >= threshold).sum().item())

            correct[j] += (train_top >= t).sum().item()
            correct[j] += (out_top < t).sum().item()
            total[j] += train_top.size(0) + out_top.size(0)

    #print(true_positives,',',false_positives,',',false_negatives)

    for j, t in enumerate(thresholds):
        accuracy = 100 * correct[j] / total[j]
        precision = true_positives[j] / (true_positives[j] + false_positives[j]) if true_positives[j] + false_positives[j] != 0 else 0
        recall = true_positives[j] / (true_positives[j] + false_negatives[j]) if true_positives[j] + false_negatives[j] !=0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))


    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
