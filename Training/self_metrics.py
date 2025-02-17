def Binary17_got_confusion_matrix(y_test, y_pred):
    import numpy as np
    import pandas as pd
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(y_test)):
        if y_test[i] == 7 and y_pred[i] == 7:
            TP += 1
        elif y_test[i] == 7 and y_pred[i] == 1:
            FN += 1
        elif y_test[i] == 1 and y_pred[i] == 7:
            FP += 1
        elif y_test[i] == 1 and y_pred[i] == 1:
            TN += 1
    return (TP, FP, FN, TN)

def Binary_got_cofusion_matrix(y_test, y_pred,labels):
    import numpy as np
    import pandas as pd
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(y_test)):
        if y_test[i] == labels[0] and y_pred[i] == labels[0]:
            TP += 1
        elif y_test[i] == labels[0] and y_pred[i] == labels[1]:
            FN += 1
        elif y_test[i] == labels[1] and y_pred[i] == labels[0]:
            FP += 1
        elif y_test[i] == labels[1] and y_pred[i] == labels[1]:
            TN += 1
    return (TP, FP, FN, TN)




def Binary_got_metrics(TP, FP, FN, TN):
    import numpy as np
    import pandas as pd
    # ## labels = [7,1] # 7 is positive and 1 is negative
    # train_label1 = int(labels[0]) # positive
    # train_label0 = int(labels[1]) # negative
    # print(f"Positive Label: {train_label1}")
    # print(f"Negative Label: {train_label0}")
    # TP = 0
    # FP = 0
    # FN = 0
    # TN = 0
    # for i in range(len(y_test)):
    #     if y_test[i] == train_label1 and y_pred[i] == train_label1:
    #         TP += 1
    #     if y_test[i] ==  train_label1 and y_pred[i] == train_label0:
    #         FN += 1
    #     if y_test[i] == train_label0 and y_pred[i] == train_label1:
    #         FP += 1
    #     if y_test[i] == train_label0 and y_pred[i] == train_label0:
    #         TN += 1
    #TP, FP, FN, TN = Binary_got_cofusion_matrix(y_train ,y_test, y_pred, labels)
    #                       Classficied Positive   Classified Negative 
    # Actual Positive             TP                   FN
    # Actual Negative             FP                   TN  
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TN: {TN}")
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    misclassification_rate = 1 - Accuracy
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    Negative_Predictive_Value = TN / (TN + FN)
    print(f"Accuracy: {Accuracy:.4f}")
    print(f"Misclassification rate: {misclassification_rate:.4f}")
    print(f"Sensitivity (Recall): {Sensitivity:.4f}")
    print(f"Specificity: {Specificity:.4f}")
    print(f"Precision: {Precision:.4f}")
    print(f"Negative Predictive Value: {Negative_Predictive_Value:.4f}")
    Gmean = np.sqrt(Sensitivity * Specificity)
    print(f"G-mean: {Gmean:.4f}")
    Fmean = (2 * Precision * Sensitivity) / (Precision + Sensitivity)
    print(f"F-measure: {Fmean:.4f}")

    # Discriminat Power X = sensitivity/(1-sepcificity) Y = specificity/(1-sensitivity)
    if Specificity == 1 or Sensitivity == 1:
        print("Discriminant Power: Infinity")
    else:
        X = Sensitivity/(1-Sensitivity)
        Y = Specificity/(1-Specificity)
        DP  =  np.sqrt(3)/np.pi * (np.log(X) + np.log(Y))
        print(f"Discriminant Power: {DP:.4f}")
    F2 = 5 * Precision * Sensitivity / (4 * Sensitivity+ Precision)
    print(f"F2-measure: {F2:.4f}")

    InvF_05 = 1.25 * Precision * Sensitivity / (0.25 * Sensitivity + Precision)
    print(f"InvF0.5-measure: {InvF_05:.4f}")

    AGF = np.sqrt(F2 * InvF_05)
    print(f"AGF: {AGF:.4f}")

    Balanced_Accuracy = (Sensitivity + Specificity) / 2
    print(f"Balanced Accuracy: {Balanced_Accuracy:.4f}")

    # Mattew's Correlation Coefficient

    MCC = (TP *TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print(f"Matthew's Correlation Coefficient: {MCC:.4f}")

    # Cohen's Kappa
    Total_Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Random_Accuracy = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / ((TP + FP + FN + TN)**2)
    Kappa = (Total_Accuracy - Random_Accuracy) / (1 - Random_Accuracy)
    print(f"Cohen's Kappa: {Kappa:.4f}")

    # Youden's Index
    Youden_Index = Sensitivity + Specificity - 1
    print(f"Youden's Index: {Youden_Index:.4f}")

    # Likelihoods Ratios

    LR_pos = Sensitivity / (1 - Specificity)
    LR_neg = (1 - Sensitivity) / Specificity

    print(f"Positive Likelihood Ratio: {LR_pos:.4f}")
    print(f"Negative Likelihood Ratio: {LR_neg:.4f}")

