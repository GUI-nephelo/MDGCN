import tensorflow as tf
import numpy as np
import time
from funcCNN import *
from BuildSPInst_A import *
from sklearn import metrics


method = "MDGCN"
data_name = 'KSC'
num_classes = 13
img_gyh = data_name+''
img_gt = data_name+'_gt'

epochs = 5000

Data = load_HSI_data(data_name)
model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh]/Data[img_gyh].max(), Data[img_gt], Data['trpos'])
sp_mean = np.array(model.sp_mean, dtype='float32')
sp_label = np.array(model.sp_label, dtype='float32')
trmask = np.matlib.reshape(np.array(model.trmask, dtype='bool'), [np.shape(model.trmask)[0], 1])
temask = np.matlib.reshape(np.array(model.temask, dtype='int32'), [np.shape(model.trmask)[0], 1])


# <tf.Tensor 'add_7:0' shape=(949, 16) dtype=float32>, 
# <tf.Tensor 'add_8:0' shape=() dtype=float32>, 
# <tf.Tensor 'Mean_3:0' shape=() dtype=float32>

# GCNmodel.concat_vec, 
# GCNmodel.loss, 
# GCNmodel.accuracy

# <Tensor("Placeholder:0", shape=(?, 1), dtype=int32>,
# <Tensor("Placeholder_1:0", shape=(?, 16), dtype=float32>

# mask
# labels




# exit()
# mask = tf.placeholder("int32", [None, 1])
# labels = tf.placeholder("float", [None, 16])

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(f'./checkpoints/{data_name}.ckpt-{epochs}.meta')
    ckpt = tf.train.get_checkpoint_state('./checkpoints/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    # graph value
    graph = tf.get_default_graph()

    get_value = lambda x : graph.get_tensor_by_name(x)


    def GCNevaluate(mask1, labels1):
        t_test = time.time()
        # with graph.as_default():
        outs_val = sess.run([get_value('add_7:0'), get_value('add_8:0'), get_value('Mean_3:0')],
            feed_dict={get_value('Placeholder_1:0'): labels1, get_value('Placeholder:0'): mask1})
                # feed_dict={labels: labels1, mask: mask1})
        return outs_val, (time.time() - t_test)

    print("labels:",sp_label.shape,sp_label.dtype)
    print("mask:",temask.shape,temask.dtype)

    test_val, test_duration = GCNevaluate(temask, sp_label)
    print(sp_label.shape, test_val[0].shape)
    y_map = sp_label.argmax(1)
    pre_map = test_val[0].argmax(1)

    OA = metrics.accuracy_score(y_map, pre_map)
    kappa = metrics.cohen_kappa_score(pre_map,y_map)
    confusion_matrix = metrics.confusion_matrix(y_map, pre_map)
    producer_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    AA = np.average(producer_acc)
    
    print("\ntest OA=", OA, ' test AA=', AA, ' kpp=', kappa, "\nproducer_acc",
          producer_acc, '\nconfusion matrix=', confusion_matrix)
    # save
    f = open('./results/' + data_name + f'_{epochs}_results.txt', 'a+')
    str_results = '\n' + method + ': ======================' \
                  + "\nOA=" + str(OA) \
                  + "\nAA=" + str(AA) \
                  + '\nkpp=' + str(kappa) \
                  + '\nacc per class:' + str(producer_acc) \
                  + "\nconfusion matrix:" + str(confusion_matrix) + "\n"

    f.write(str_results)
    f.close()
