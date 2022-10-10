# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN
import one_hot_encoding

def main():
    for number in range(6, 7):
        cnn = CNN()
        cnn.eval()
        name = 'model' + str(number) + '.pkl'
        cnn.load_state_dict(torch.load(name))
        #print("load cnn net.")

        test_dataloader = my_dataset.get_test_data_loader()

        correct = 0
        corlen = 0
        total = 0
        true_len = [0, 0, 0, 0, 0]
        label_len = [0, 0, 0, 0, 0]
        for i, (images, labels) in enumerate(test_dataloader):
            image = images
            vimage = Variable(image)
            predict_label = cnn(vimage)
            #print(np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy()))

            c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c4 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 4 * captcha_setting.ALL_CHAR_SET_LEN:5 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c5 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 5 * captcha_setting.ALL_CHAR_SET_LEN:6 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c6 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 6 * captcha_setting.ALL_CHAR_SET_LEN:7 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c7 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 7 * captcha_setting.ALL_CHAR_SET_LEN:8 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            predict_label = '%s%s%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5, c6, c7)
            predict_label = predict_label.replace(" ", "")
            true_label = one_hot_encoding.decode(labels.numpy()[0])
            true_label = true_label.replace("_", "")
            #print(predict_label, " ", true_label)
            total += labels.size(0)
            if(predict_label == true_label):
                correct += 1
                corlen += 1
                true_len[len(true_label) - 4] += 1
                
            #if(total%200==0):
            #    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
                #break
            #if len(predict_label) == len(true_label):
                #corlen += 1
                #true_len[len(true_label) - 4] += 1
            label_len[len(true_label) - 4] += 1
        print('Test Accuracy of the %d model on the %d test images: %f %%' % (number, total, 100 * correct / total))
        print(corlen / total * 100)
        for i in range(len(label_len)):
            print('Accuracy of len = %d is %f %%' % (i + 4, true_len[i] / label_len[i] * 100))

if __name__ == '__main__':
    main()


