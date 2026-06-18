import tensorflow as tf
import numpy as np
import sys, os
import random
import datetime
import traceback
from evaluate import evaluate

from module import Net
from utils import simplex_project, log, save_config, load_data, validation_split

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_norm', 1, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('n_in', 5, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 4, """Number of output layers. """)
tf.app.flags.DEFINE_integer('dim_in', 32, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 128, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_float('p_coef_y', 1.0, """ Default 1: Outcome Regression - Loss FUN L_R in Eq.(8). """)
tf.app.flags.DEFINE_float('p_coef_mu', 5, """Hyper-parameter mu: Representation orthogonal regularizer weight.""")
tf.app.flags.DEFINE_float('p_coef_lambda', 1e-3, """Hyper-parameter lambda: representation-layer weight decay coefficient.""")
tf.app.flags.DEFINE_float('p_coef_mi', 1.0, """Hyper-parameter for Mutual Information total loss (controls all MI terms).""")
# Training Configurations
tf.app.flags.DEFINE_integer('seed', 1, """Random Seed. """)
tf.app.flags.DEFINE_integer('experiments', 10, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 300, """Number of single-stage training iterations.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('loss', 'log', """Type of loss function to use: 'log' for binary outcomes, 'l1' or 'l2' for continuous outcomes.""")
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_integer('ycf_result', 1, """The exits of ycf. """)
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 30, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)

tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_string('outdir', 'results/example_jobs/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '/home/student1/projects/ORTHO/data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'jobs_DW_bin.new.10.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'jobs_DW_bin.new.10.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)

CONFIG_FLAGS = [
    'batch_norm', 'batch_size', 'data_test', 'datadir', 'dataform', 'decay',
    'dim_in', 'dim_out', 'dropout_in', 'dropout_out', 'experiments', 'loss',
    'lrate', 'lrate_decay', 'n_in', 'n_out', 'nonlin', 'normalization',
    'optimizer', 'outdir', 'output_csv', 'output_delay', 'p_coef_lambda',
    'p_coef_mi', 'p_coef_mu', 'p_coef_y', 'pred_output_delay',
    'rep_weight_decay', 'seed', 'split_output',
    'iterations', 'val_part', 'varsel',
    'weight_init', 'ycf_result'
]


NUM_ITERATIONS_PER_DECAY = 100

def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp, outdir):
    """Train one experiment with the fixed single-stage schedule."""

    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)
    p_treated = np.mean(D['t'][I_train,:])
    t_train = D['t'][I_train,:]
    yff = D['yf'][I_train,:]
    yff_0 = yff[t_train[:,0] < 0.5,:]
    yff_1 = yff[t_train[:,0] > 0.5,:]
    yff_0_median = np.median(yff_0)
    yff_1_median = np.median(yff_1)

    dict_factual = {
        CFR.x: D['x'][I_train,:], CFR.t: D['t'][I_train,:], CFR.y_: D['yf'][I_train,:],
        CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.p_t: p_treated,
        CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median,
    }

    if FLAGS.val_part > 0:
        dict_valid = {
            CFR.x: D['x'][I_valid,:], CFR.t: D['t'][I_valid,:], CFR.y_: D['yf'][I_valid,:],
            CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.p_t: p_treated,
            CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median,
        }

    if D['HAVE_TRUTH']:
        dict_cfactual = {
            CFR.x: D['x'][I_train,:], CFR.t: 1-D['t'][I_train,:], CFR.y_: D['ycf'][I_train,:],
            CFR.do_in: 1.0, CFR.do_out: 1.0,
            CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median,
        }

    sess.run(tf.compat.v1.global_variables_initializer())

    preds_train = []
    preds_test = []

    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    valid_imb_c = np.nan; valid_imb_i = np.nan; valid_imb_a = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb, valid_imb_c, valid_imb_i, valid_imb_a = sess.run(
            [CFR.val_loss, CFR.pred_loss, CFR.imb_dist, CFR.imb_dist_C, CFR.imb_dist_I, CFR.imb_dist_A],
            feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    best_valid_metric = float('inf')
    best_iteration = 0
    patience = 50
    patience_counter = 0


    total_iterations = FLAGS.iterations

    for i in range(total_iterations):

        if FLAGS.batch_size == 0:
            batch_indices = random.sample(range(0, n_train), n_train)
        else:
            batch_indices = random.sample(range(0, n_train), FLAGS.batch_size)

        x_batch = D['x'][I_train, :][batch_indices, :]
        t_batch = D['t'][I_train, :][batch_indices]
        y_batch = D['yf'][I_train, :][batch_indices]
        if FLAGS.ycf_result == 1:
            yc_batch = D['ycf'][I_train, :][batch_indices]

        if not objnan:
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                                            CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                            CFR.p_t: p_treated, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})

        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})

        if i % FLAGS.output_delay == 0 or i==total_iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual)

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            valid_imb_c = np.nan; valid_imb_i = np.nan; valid_imb_a = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb, valid_imb_c, valid_imb_i, valid_imb_a = sess.run(
                    [CFR.val_loss, CFR.pred_loss, CFR.imb_dist, CFR.imb_dist_C, CFR.imb_dist_I, CFR.imb_dist_A],
                    feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = ('Iter-' + str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f') \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)
            
            if FLAGS.val_part > 0:
                loss_str += ',\tImbC: %.2g,\tImbI: %.2g,\tImbA: %.2g' % (valid_imb_c, valid_imb_i, valid_imb_a)

            if FLAGS.loss == 'log':
                y_pred_discrete = sess.run(CFR.output_discrete, feed_dict={CFR.x: x_batch, \
                    CFR.t: t_batch, CFR.y_: y_batch, CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                y_pred_discrete = 1.0*(y_pred_discrete > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred_discrete)))
                loss_str += ',\tAcc: %.2f%%' % acc
                if FLAGS.ycf_result == 1:
                    yc_pred_discrete = sess.run(CFR.output_discrete, feed_dict={CFR.x: x_batch, \
                                                              CFR.t: 1 - t_batch, CFR.y_: yc_batch, CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                    yc_pred_discrete = 1.0 * (yc_pred_discrete > 0.5)
                    cacc = 100 * (1 - np.mean(np.abs(yc_batch - yc_pred_discrete)))
                    loss_str += ',\tcAcc: %.2f%%' % cacc

            log(logfile, loss_str)

            if FLAGS.val_part > 0:
                if FLAGS.ycf_result == 1:
                    early_stop_metric = valid_f_error
                    metric_name = "Valid PEHE"
                else:
                    early_stop_metric = valid_obj
                    metric_name = "Valid Obj"
                
                if not np.isnan(early_stop_metric):
                    if early_stop_metric < best_valid_metric:
                        best_valid_metric = early_stop_metric
                        best_iteration = i
                        patience_counter = 0
                        log(logfile, '  → New best %s: %.4f at iteration %d' % (metric_name, best_valid_metric, i))
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        log(logfile, '\n*** Early Stopping triggered at iteration %d ***' % i)
                        log(logfile, '*** Best %s: %.4f at iteration %d ***' % (metric_name, best_valid_metric, best_iteration))
                        log(logfile, '*** Patience: %d iterations without improvement ***\n' % patience)
                        break

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True
                exit()

        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==total_iterations-1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: D['t'], CFR.y_: D['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            if FLAGS.ycf_result == 1:
                y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                    CFR.t: 1-D['t'], CFR.y_: D['ycf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            else:
                y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                    CFR.t: 1-D['t'], CFR.y_: D['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: D_test['t'], CFR.y_: D_test['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                if FLAGS.ycf_result == 1:
                    y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                        CFR.t: 1-D_test['t'], CFR.y_: D_test['ycf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                else:
                    y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                        CFR.t: 1-D_test['t'], CFR.y_: D_test['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

    w_I, w_C, w_A, w_out, w_pred = sess.run([CFR.weights_in_I, CFR.weights_in_C, CFR.weights_in_A,
                                             CFR.weights_out, CFR.weights_pred], feed_dict={CFR.x: D['x']})
    if os.path.exists(outdir + 'w/'):
        pass
    else:
        os.makedirs(outdir + 'w/')
    npzfile_w = outdir + 'w/w_' + str(999)
    log(logfile, npzfile_w)
    np.savez(npzfile_w, w_I=w_I, w_C=w_C, w_A=w_A, w_out=w_out, w_pred=w_pred)

    return losses, preds_train, preds_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '':
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    save_config(outdir+'config.txt', CONFIG_FLAGS)

    log(logfile, 'Training with hyperparameters: p_coef_y=%.2g, p_coef_mu=%.2g, p_coef_lambda=%.2g, p_coef_mi=%.2g, iterations=%d. ' % (
        FLAGS.p_coef_y, FLAGS.p_coef_mu, FLAGS.p_coef_lambda, FLAGS.p_coef_mi, FLAGS.iterations))

    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d]' % (D['dim']))

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    log(logfile, 'Defining graph...\n')
    dims = [D['n'], D['dim'], FLAGS.dim_in, FLAGS.dim_out]
    CFR = Net(dims, FLAGS)

    train_step_counter = tf.compat.v1.Variable(0, trainable=False, name='train_step')

    learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.lrate, train_step_counter, \
                                                         NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    def make_optimizer(learning_rate):
        if FLAGS.optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer(learning_rate)
        if FLAGS.optimizer == 'GradientDescent':
            return tf.train.GradientDescentOptimizer(learning_rate)
        if FLAGS.optimizer == 'Adam':
            return tf.compat.v1.train.AdamOptimizer(learning_rate)
        return tf.compat.v1.train.RMSPropOptimizer(learning_rate, FLAGS.decay)

    optimizer = make_optimizer(learning_rate)

    D_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='representation')
    O_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='output')

    MI_vars = []
    for mi_name in ['mi_it', 'mi_iy', 'mi_ct', 'mi_cy', 'mi_at', 'mi_ay']:
        MI_vars += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=mi_name)

    train_vars = D_vars + O_vars + MI_vars

    train_step = optimizer.minimize(CFR.tot_loss, global_step=train_step_counter, var_list=train_vars)

    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    n_experiments = FLAGS.experiments
    for i_exp in range(1, n_experiments+1):
        log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))
        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        losses, preds_train, preds_test = \
            train(CFR, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp, outdir)

        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'results_'+timestamp+'/'
    os.makedirs(outdir)

    try:
        run(outdir)
        # 根据outdir自动选择评估目录
        eval_dir = outdir
        print(f'\n开始评估实验结果: {eval_dir}')
        evaluate(eval_dir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.compat.v1.app.run()
