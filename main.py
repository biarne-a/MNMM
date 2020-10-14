import os
import pickle
import numpy as np
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import pandas as pd
from dataset_preparator import DataSetPreparator
from multinomial_expectation_maximizer import MultinomialExpectationMaximizer, \
    IndividualMultinomialExpectationMaximizer


SAVE_DIR = 'run_with_gamma'


def get_individual_alpha_test(best_alpha, test_household_ids, train_household_ids):
    alpha_df = pd.DataFrame(best_alpha, index=train_household_ids)
    filtered_alpha_df = alpha_df.groupby(alpha_df.index).apply(lambda g: g.iloc[0])
    alpha_test = np.vstack([filtered_alpha_df.loc[household_id].values for household_id in test_household_ids])
    return alpha_test


def process_with_k_mixtures(args):
    k, X_train, X_test, train_household_ids, test_household_ids, individual_mode = args
    print('###################################### Experiment with %i mixture components ######################################' % k)

    best_train_loss, log_likelihood, bic, best_alpha, best_beta = pickle.load(open('best_params_%i.p' % k, 'rb'))

    if individual_mode:
        model = IndividualMultinomialExpectationMaximizer(k, best_alpha, best_beta, train_household_ids,
                                                          restarts=10, rtol=1e-4)
        best_train_loss, best_alpha, best_beta, best_gamma = model.fit(X_train)
        alpha_test = get_individual_alpha_test(best_alpha, test_household_ids, train_household_ids)
    else:
        model = MultinomialExpectationMaximizer(k, restarts=10, rtol=1e-4)
        best_train_loss, best_alpha, best_beta, best_gamma = model.fit(X_train)
        alpha_test = best_alpha

    log_likelihood = model.compute_log_likelihood(X_test, alpha_test, best_beta)
    bic = model.compute_bic(X_test, best_alpha, best_beta, log_likelihood)
    icl_bic = model.compute_icl_bic(bic, best_gamma)

    print('log likelihood for k=%i : %f' % (k, log_likelihood))
    print('bic for k=%i : %f' % (k, bic))
    print('icl bic for k=%i : %f' % (k, icl_bic))
    pickle.dump((best_train_loss, log_likelihood, bic, best_alpha, best_beta, best_gamma),
                open(os.path.join(SAVE_DIR, 'best_params_%i.p' % k), 'wb'))


def load_data(from_disk):
    if from_disk:
        train_grocery_df, test_grocery_df, X_train, X_test, train_household_ids, test_household_ids = \
            pickle.load(open(os.path.join(SAVE_DIR, 'data.p'), 'rb'))
        return X_train, X_test, train_household_ids, test_household_ids

    transactions_filepath = 'data/transaction_data.csv'
    products_filepath = 'data/product.csv'

    train_grocery_df, test_grocery_df, train_counts_df, test_counts_df = DataSetPreparator.prepare(
        transactions_filepath, products_filepath)
    train_household_ids = train_counts_df.index.droplevel(level=1)
    test_household_ids = test_counts_df.index.droplevel(level=1)
    X_train, X_test = train_counts_df.values, test_counts_df.values

    unknown_ids = set(test_household_ids).difference(set(train_household_ids))
    X_test = X_test[~test_household_ids.isin(unknown_ids)]
    test_household_ids = test_household_ids[~test_household_ids.isin(unknown_ids)]

    data = (
    train_grocery_df, test_grocery_df, train_counts_df, test_counts_df, train_household_ids, test_household_ids)
    pickle.dump(data, open(os.path.join(SAVE_DIR, 'data.p'), 'wb'))

    return X_train, X_test, train_household_ids, test_household_ids


if __name__ == '__main__':
    X_train, X_test, train_household_ids, test_household_ids = load_data(from_disk=True)

    individual_mode = False
    Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    params = [(k, X_train, X_test, train_household_ids, test_household_ids, individual_mode) for k in Ks]
    p_counts = cpu_count()
    with ThreadPool(processes=p_counts) as pool:
         pool.map(process_with_k_mixtures, params)
