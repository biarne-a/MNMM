import pickle
import numpy as np
import pandas as pd


class DataSetPreparator:
    @staticmethod
    def _load_full_dataframe(products_filepath, transactions_filepath):
        transactions_df = pd.read_csv(transactions_filepath)
        products_df = pd.read_csv(products_filepath)
        products_df.set_index('PRODUCT_ID', inplace=True)
        full_df = transactions_df.join(products_df, on='PRODUCT_ID', how='inner')
        return full_df

    @staticmethod
    def _prepare_count_data(grocery_df):
        baskets_data = grocery_df.groupby(['household_key', 'BASKET_ID'])['COMMODITY_DESC'].value_counts()
        baskets_data_df = pd.DataFrame(data=baskets_data.values, index=baskets_data.index,
                                       columns=['Count']).reset_index()
        counts_df = baskets_data_df.pivot(index=['household_key', 'BASKET_ID'], columns=['COMMODITY_DESC'],
                                          values=['Count'])['Count']
        counts_df.fillna(0, inplace=True)
        return counts_df

    @staticmethod
    def prepare(transactions_filepath, products_filepath):
        full_df = DataSetPreparator._load_full_dataframe(products_filepath, transactions_filepath)
        grocery_df = full_df.loc[full_df['DEPARTMENT'] == 'GROCERY'].copy()
        train_grocery_df, test_grocery_df = DataSetPreparator._split_train_test(grocery_df)
        train_counts_df = DataSetPreparator._prepare_count_data(train_grocery_df)
        test_counts_df = DataSetPreparator._prepare_count_data(test_grocery_df)
        return train_grocery_df, test_grocery_df, train_counts_df, test_counts_df

    @staticmethod
    def _split_train_test(grocery_df):
        grocery_df.sort_values(['WEEK_NO', 'DAY', 'TRANS_TIME'], inplace=True)
        grocery_df.reset_index(inplace=True)
        unique_basket_ids = grocery_df['BASKET_ID'].unique()
        nb_train = int(len(unique_basket_ids) * 0.8)
        train_grocery_df = grocery_df[grocery_df['BASKET_ID'].isin(unique_basket_ids[:nb_train])]
        test_grocery_df = grocery_df[grocery_df['BASKET_ID'].isin(unique_basket_ids[nb_train:])]
        return train_grocery_df, test_grocery_df


class DataSetPreparator2:
    @staticmethod
    def _load_full_dataframe():
        # data_df = pd.read_excel('/home/adrien/Téléchargements/online_retail_II.xlsx')
        data_df = pickle.load(open('/home/adrien/Téléchargements/online_retail_II.p', 'rb'))
        return data_df.dropna()

    @staticmethod
    def _filter_out_non_frequent_products(data_df, min_product_count=10):
        product_counts = data_df['Description'].value_counts()
        return data_df[data_df['Description'].isin(product_counts[product_counts > min_product_count].index)]

    @staticmethod
    def _prepare_count_data(data_df):
        baskets_data = data_df.groupby(['Customer ID', 'Invoice'])['Description'].value_counts()
        baskets_data_df = pd.DataFrame(data=baskets_data.values, index=baskets_data.index,
                                       columns=['Count']).reset_index()
        counts_df = baskets_data_df.pivot(index=['Customer ID', 'Invoice'], columns=['Description'],
                                          values=['Count'])['Count']
        counts_df.fillna(0, inplace=True)
        return counts_df

    @staticmethod
    def prepare():
        data_df = DataSetPreparator2._load_full_dataframe()
        data_df = DataSetPreparator2._filter_out_non_frequent_products(data_df)
        train_data_df, test_data_df = DataSetPreparator2._split_train_test(data_df)
        train_counts_df = DataSetPreparator2._prepare_count_data(train_data_df)
        test_counts_df = DataSetPreparator2._prepare_count_data(test_data_df)

        print('train_counts_df.shape = %s' % str(train_counts_df.shape))
        print('test_counts_df.shape = %s' % str(test_counts_df.shape))
        train_customer_ids = train_counts_df.index.droplevel(level=1)
        test_customer_ids = test_counts_df.index.droplevel(level=1)
        return train_customer_ids, test_customer_ids, train_counts_df.values, test_counts_df.values

    @staticmethod
    def _split_train_test(data_df):
        unique_customer_ids = data_df['Customer ID'].unique()
        np.random.shuffle(unique_customer_ids)
        nb_train = int(len(unique_customer_ids) * 0.8)
        train_data_df = data_df[data_df['Customer ID'].isin(unique_customer_ids[:nb_train])]
        test_data_df = data_df[data_df['Customer ID'].isin(unique_customer_ids[nb_train:])]
        return train_data_df, test_data_df
