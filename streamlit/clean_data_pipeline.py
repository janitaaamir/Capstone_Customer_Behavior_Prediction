import pandas as pd
import re
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
from datetime import datetime, timedelta
import math
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None


def load_and_merge_data(customer_orders_df, census_data_path):

    customer_orders_df = customer_orders_df.copy()
    customer_orders_df['billing_postcode'] = customer_orders_df['billing_postcode'].astype(str).fillna("")
    customer_orders_df['shipping_postcode'] = customer_orders_df['shipping_postcode'].astype(str).fillna("")

    census_data = pd.read_csv(census_data_path, dtype={'zip': str})

    customer_orders_df['zip'] = customer_orders_df['billing_postcode'].apply(lambda x: x[:5] if len(x) >= 5 else "")

    merged_data = customer_orders_df.merge(
        census_data[['zip', 'median_household_income', 'moe_median_household_income',
                     'mean_household_income', 'moe_mean_household_income']],
        on='zip',
        how='left'
    )

    return merged_data


def fetch_gender_data():
    gender_data_path = "gender_by_name.csv" 

    if not os.path.exists(gender_data_path):
        raise FileNotFoundError(
            f"Gender dataset not found. Please download and save it as '{gender_data_path}'.")

    gender_df = pd.read_csv(gender_data_path)
    
    gender_df['Name'] = gender_df['Name'].str.strip().str.lower()

    gender_df = gender_df.sort_values(by=['Name', 'Probability'], ascending=[True, False]).drop_duplicates(subset='Name')

    gender_lookup = gender_df.set_index('Name')[['Gender', 'Probability']].to_dict(orient='index')

    return gender_lookup



def add_gender_info(df, gender_lookup):
    def get_gender_info(name):
        info = gender_lookup.get(str(name).lower(), {"Gender": "Unknown", "Probability": None})
        return info['Gender'], info['Probability']

    df[['gender', 'probability_of_gender']] = df['billing_first_name'].apply(
        lambda name: pd.Series(get_gender_info(name)))

    return df


def parse_date(date_str):
    dt = pd.to_datetime(date_str)
    return pd.Series({
        'order_day': dt.day,
        'order_month': dt.month,
        'order_year': dt.year,
        'order_time': dt.strftime('%H:%M:%S')
    })


def standardize_phone(phone):
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10:
        return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return digits


def standardize_postcode(postcode):
    postcode = str(postcode).strip()
    if '-' in postcode:
        main, extra = postcode.split('-', 1)
        return pd.Series([main, extra[:4]])
    else:
        return pd.Series([postcode, None])

state_abbreviations = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
        'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA',
        'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
        'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
        'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
        'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
        'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
        'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
        'district of columbia': 'DC', 'dc': 'DC'
}

def standardize_state(state):
    state = str(state).strip().lower()
    return state_abbreviations.get(state, state).upper()



def parse_items(df):

    def parse_data(items_str, prefix):
        if pd.isna(items_str):
            return {}
        items = items_str.split(';')
        parsed_data = {}
        for i, item in enumerate(items):
            details = item.split('|')
            for detail in details:
                if ':' in detail:
                    key, value = detail.split(':', 1)
                    parsed_data[f'{prefix}_{i + 1}_{key}'] = value
        return parsed_data

    df = pd.concat([df.drop(columns=['line_items', 'shipping_items', 'coupon_items']),
                    df['line_items'].apply(lambda x: pd.Series(parse_data(x, 'item'))),
                    df['shipping_items'].apply(lambda x: pd.Series(parse_data(x, 'shipping'))),
                    df['coupon_items'].apply(lambda x: pd.Series(parse_data(x, 'coupon')))], axis=1)

    return df


def clean_income_data(df):

    def clean_income_column(column):
        return (
            column.replace({',': '', r'\+': '', '-': '', 'N': ''}, regex=True)
            .replace('', np.nan)
            .astype(float)
        )

    df['median_household_income'] = clean_income_column(df['median_household_income'])
    df['mean_household_income'] = clean_income_column(df['mean_household_income'])
    df['is_income_censored'] = (df['moe_median_household_income'] == "***") | (df['moe_mean_household_income'] == "***")

    df['moe_median_household_income'] = df['moe_median_household_income'].replace(['***', '**', '-', 'N'],
                                                                                  np.nan).astype(float)
    df['moe_mean_household_income'] = df['moe_mean_household_income'].replace(['***', '**', '-', 'N'], np.nan).astype(
        float)

    return df


def extract_area_code(phone):
    if pd.isna(phone) or phone.strip() == '':
        return 'UNKNOWN'
    match = re.search(r'\((\d{3})\)', phone)
    return match.group(1) if match else 'UNKNOWN'


def categorize_email(df):

    def get_category(domain):
        personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com', 'aol.com', 'icloud.com']
        education_domains = ['.edu']
        corporate_domains = ['.com', '.net', '.org']

        if domain == 'UNKNOWN':
            return 'UNKNOWN'
        elif any(domain.endswith(edu) for edu in education_domains):
            return 'EDUCATION'
        elif domain in personal_domains:
            return 'PERSONAL'
        elif any(domain.endswith(corp) for corp in corporate_domains):
            return 'CORPORATE'
        else:
            return 'OTHER'

    df['email_domain'] = df['billing_email'].apply(lambda email: email.split('@')[1] if pd.notna(email) else 'UNKNOWN')
    df['email_category'] = df['email_domain'].apply(get_category)

    return df

def categorize_billing_company(df):
    
    df['billing_company'] = df['billing_company'].astype(str).fillna("").str.strip()

    df['has_billing_company'] = df['billing_company'] != ""

    def get_company_type(name):
        if name in ['unknown', '', 'optional', 'title (optional)']:
            return 'UNKNOWN'
        elif 'retired' in name.lower():
            return 'RETIREMENT'
        elif 'guest' in name.lower() or 'self employed' in name.lower():
            return 'PERSONAL'
        else:
            return 'BUSINESS'

    df['billing_company_category'] = df['billing_company'].apply(get_company_type)

    return df


def categorize_shipping_company(df):
    
    df['shipping_company'] = df['shipping_company'].astype(str).fillna("").str.strip()

    df['has_shipping_company'] = df['shipping_company'] != ""

    def get_company_type(name):
        if name in ['unknown', '', 'optional', 'title (optional)']:
            return 'UNKNOWN'
        elif 'retired' in name.lower():
            return 'RETIREMENT'
        elif 'guest' in name.lower() or 'self employed' in name.lower():
            return 'PERSONAL'
        else:
            return 'BUSINESS'

    df['shipping_company_category'] = df['shipping_company'].apply(get_company_type)

    return df



def calculate_repeat_purchases(df):

    df['customer_identifier'] = df['billing_first_name'].str.lower() + '_' + \
                                df['billing_last_name'].str.lower() + '_' + \
                                df['billing_zip_main'].astype(str)

    purchase_counts = df['customer_identifier'].value_counts()
    df['purchase_count'] = df['customer_identifier'].map(purchase_counts)
    df['repeat_purchase'] = df['purchase_count'].apply(lambda x: 1 if x > 1 else 0)

    repeat_count = df['repeat_purchase'].sum()

    customer_id_counts = df['customer_id'].value_counts()
    total_repeat_purchases = customer_id_counts[customer_id_counts > 1].sum()

    return df


def sort_and_identify_repeat_customers(df):

    df['person_id'] = (
            df['billing_first_name'].str.strip().str.lower() + " " +
            df['billing_last_name'].str.strip().str.lower() + " " +
            df['billing_zip_main'].fillna("").astype(str).str.lower()
    )

    df['order_datetime'] = pd.to_datetime(
        df[['order_year', 'order_month', 'order_day', 'order_time']]
        .astype(str)
        .agg('-'.join, axis=1)
    )

    df = df.sort_values(by=['person_id', 'order_datetime'])
    df['repeat_customer'] = 0
    df.loc[df.duplicated(subset=['person_id']), 'repeat_customer'] = 1

    # df['repeat_customer'] = df.groupby('person_id')['person_id'] \
    #     .transform(lambda x: [1] * (len(x) - 1) + [0] if len(x) > 1 else [0])

    return df

def drop_cols(df):
    drop_cols = [
        'order_id', 'order_number', 'order_number_formatted', 'status', 'customer_id',
        'billing_company', 'billing_phone', 'billing_address_1',
        'billing_address_2', 'shipping_first_name', 'shipping_last_name', 'shipping_address_1', 'shipping_address_2',
        'shipping_company', 'customer_note', 'fee_items', 'tax_items', 'order_notes', 'shipment_tracking',
        'probability_of_gender', 'zip', 'billing_zip_extra', 'shipping_zip_extra', 'item_1_id', 'item_2_id',
        'item_3_id', 'item_4_id', 'item_5_id', 'item_6_id', 'item_7_id', 'item_1_name',
        'item_2_name', 'item_3_name', 'item_4_name', 'item_5_name', 'item_6_name', 'item_7_name',
        'item_1_meta', 'item_2_meta', 'item_3_meta', 'item_4_meta', 'item_5_meta', 'item_6_meta',
        'item_7_meta', 'shipping_1_method_id', 'coupon_1_code', 'coupon_1_description', 'customer_identifier',
        'purchase_count', 'repeat_purchase', 'shipping_1_method_title',
        'item_1_subtotal_tax', 'item_2_subtotal_tax', 'item_3_subtotal_tax', 'item_4_subtotal_tax',
        'item_5_subtotal_tax', 'item_6_subtotal_tax', 'item_7_subtotal_tax', 'item_1_total_tax', 'item_2_total_tax',
        'item_3_total_tax', 'item_4_total_tax', 'item_5_total_tax', 'item_6_total_tax', 'item_7_total_tax',
        'item_1_total', 'item_2_total', 'item_3_total', 'item_4_total', 'item_5_total', 'item_6_total', 'item_7_total',
        'item_1_refunded', 'item_2_refunded', 'item_3_refunded', 'item_4_refunded', 'item_5_refunded',
        'item_6_refunded',
        'item_7_refunded', 'item_1_refunded_qty', 'item_2_refunded_qty', 'item_3_refunded_qty', 'item_4_refunded_qty',
        'item_5_refunded_qty', 'item_6_refunded_qty', 'item_7_refunded_qty',
        'shipping_tax_total', 'fee_total', 'fee_tax_total', 'tax_total', 'refunds', 'item_1_product_id',
        'item_2_product_id', 'item_3_product_id', 'item_4_product_id', 'item_5_product_id',
        'item_6_product_id', 'item_7_product_id', 'shipping_company_category'
    ]

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    if 'order_time' in df.columns:
        df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S', errors='coerce')
        df['order_hour'] = df['order_time'].dt.hour
        df['order_minute'] = df['order_time'].dt.minute
        df.drop(columns=['order_time'], inplace=True, errors='ignore')

    drop_cols_extra = ['billing_city', 'shipping_city', 'billing_city_frequency', 'shipping_city_frequency']
    
    df.drop(columns=drop_cols_extra, inplace=True, errors='ignore')

    return df

def transform_item_features(df):
    sku_columns = [col for col in df.columns if col.endswith('_sku')]
    subtotal_columns = [col.replace('_sku', '_subtotal') for col in sku_columns if col.replace('_sku', '_subtotal') in df.columns]
    quantity_columns = [col.replace('_sku', '_quantity') for col in sku_columns if col.replace('_sku', '_quantity') in df.columns]

    if not sku_columns:
        return df  

    for col in sku_columns:
        df[col] = df[col].astype(str)

    for col in subtotal_columns + quantity_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    flattened_data = []
    for index, row in df.iterrows():
        for sku_col, subtotal_col, quantity_col in zip(sku_columns, subtotal_columns, quantity_columns):
            sku = row[sku_col]
            subtotal = row[subtotal_col] if subtotal_col in df.columns else 0
            quantity = row[quantity_col] if quantity_col in df.columns else 0

            if sku and sku != "nan":  
                flattened_data.append({'index': index, 'SKU': sku, 'Subtotal': subtotal, 'Count': quantity})

    if flattened_data:
        flattened_df = pd.DataFrame(flattened_data)

        count_data = flattened_df.pivot_table(index='index', columns='SKU', values='Count', aggfunc='sum', fill_value=0)
        subtotal_data = flattened_df.pivot_table(index='index', columns='SKU', values='Subtotal', aggfunc='sum',
                                                 fill_value=0)

        count_data.columns = [f'{col}_count' for col in count_data.columns]
        subtotal_data.columns = [f'{col}_subtotal' for col in subtotal_data.columns]

        final_df = pd.concat(
            [df.drop(columns=sku_columns + subtotal_columns + quantity_columns, errors='ignore'), count_data, subtotal_data],
            axis=1
        )
        return final_df
    else:
        return df


def add_past_order_info(data):

    data['order_datetime'] = pd.to_datetime(data['order_datetime'])

    data = data.sort_values(by=['person_id', 'order_datetime']).reset_index(drop=True)

    time_since_last_purchase = []
    time_between_purchases = []
    time_since_first_purchase = []
    number_of_past_orders = []
    order_frequency_90_days = []
    average_order_value = []
    total_order_value = []
    std_order_value = []
    distinct_product_count = []
    most_frequent_product = []
    repeat_purchase_count = []

    for person_id, group in data.groupby('person_id'):
        previous_orders = []
        previous_order_dates = []
        previous_order_values = []
        previous_products = []

        for index, row in group.iterrows():
            if previous_orders:
                time_since = (row['order_datetime'] - previous_order_dates[-1]).days
                time_since_last_purchase.append(time_since)

                time_between_purchases.append(sum([
                    (previous_order_dates[i] - previous_order_dates[i - 1]).days
                    for i in range(1, len(previous_order_dates))
                ]) / len(previous_order_dates) if len(previous_order_dates) > 1 else time_since)

                time_since_first_purchase.append((row['order_datetime'] - previous_order_dates[0]).days)
            else:
                time_since_last_purchase.append(None)
                time_between_purchases.append(None)
                time_since_first_purchase.append(None)

            number_of_past_orders.append(len(previous_orders))

            recent_orders = [d for d in previous_order_dates if row['order_datetime'] - d <= timedelta(days=90)]
            order_frequency_90_days.append(len(recent_orders))

            if previous_order_values:
                average_order_value.append(sum(previous_order_values) / len(previous_order_values))
                total_order_value.append(sum(previous_order_values))
                std_order_value.append(pd.Series(previous_order_values).std())
            else:
                average_order_value.append(None)
                total_order_value.append(None)
                std_order_value.append(None)

            distinct_product_count.append(len(set(previous_products)))
            most_frequent_product.append(pd.Series(previous_products).mode()[0] if previous_products else None)

            repeat_purchase_count.append(len(previous_orders))

            previous_orders.append(row['order_total'])
            previous_order_dates.append(row['order_datetime'])
            previous_order_values.append(row['order_total'])
            product_cols = [col for col in data.columns if col.endswith('_count')]
            for col in product_cols:
                if row[col] > 0:
                    previous_products.append(col)

    data['time_since_last_purchase'] = time_since_last_purchase
    data['time_between_purchases'] = time_between_purchases
    data['time_since_first_purchase'] = time_since_first_purchase
    data['number_of_past_orders'] = number_of_past_orders
    data['order_frequency_90_days'] = order_frequency_90_days
    data['average_order_value'] = average_order_value
    data['total_order_value'] = total_order_value
    data['std_order_value'] = std_order_value
    data['distinct_product_amount'] = distinct_product_count
    data['most_frequent_product'] = most_frequent_product
    data['repeat_purchase_amount'] = repeat_purchase_count

    for col in ['billing_city', 'shipping_city']:
        city_counts = data[col].value_counts().to_dict()
        data[f'{col}_frequency'] = data[col].map(city_counts)

    return data

def remove_item_subtotal(df):
    df_filtered = df[[col for col in df.columns if not col.endswith('_subtotal')]]

    return df_filtered

def sku_condense(df):
    sku_mappings = {
        "BSH10E-1000": ["B10-1000", "SH10E-1000"],
        "BSH10-1000": ["B10-1000", "SH10-1000"],
        "CSH6E-1000-1": ["N6-1000", "SH6E-1000"],
        "CSH6E-1000": ["C6-1000", "SH6E-1000"],
        "CSH6-1000": ["C6-1000", "SH6-1000"],
        "CSH8E-1000": ["C8-1000", "SH8E-1000"],
        "CSH8-1000": ["C8-1000", "SH8-1000"],
        "NSH6-1000": ["N6-1000", "SHN6-1000"],
        #"NSH6E-1000": ["N6-1000", "SHN6E-1000"], #does not exist in old or new data yet but possible
        "PSH4E-1000": ["P4-1000", "SH4E-1000"],
        "PSH4-1000": ["P4-1000", "SH4-1000"],
        "SSH10E-1000": ["S10-1000", "SH10E-1000"],
        "SSH10-1000": ["S10-1000", "SH10-1000"],
        "ES3-1000": ["C8-1000", "P4-1000", "B10-1000"],
        "ESH3-1000": ["C8-1000", "P4-1000", "B10-1000", "SH8-1000", "SH4-1000", "SH10-1000"],
        "FF7-1000": ["C8-1000", "C6-1000", "B10-1000", "S10-1000", "P4-1000", "KB1-1000", "CK2-1000"],
        "FF8-1000": ["C8-1000", "C6-1000", "B10-1000", "S10-1000", "P4-1000", "N6-1000", "KB1-1000", "CK2-1000"],
        "SS2-1000": ["C8-1000", "P4-1000"]
    }


    count_columns = [col for col in df.columns if col.endswith('_count')]

    consolidation_examples = {}

    for old_sku, new_skus in sku_mappings.items():
        old_sku_count_col = f"{old_sku}_count"

        if old_sku_count_col in df.columns:
            original_count = df[old_sku_count_col].sum()
            consolidation_examples[old_sku] = {"original_count": original_count, "distributed_to": {}}

            for new_sku in new_skus:
                new_sku_count_col = f"{new_sku}_count"

                if new_sku_count_col not in df.columns:
                    df[new_sku_count_col] = 0

                df[new_sku_count_col] += df[old_sku_count_col]

                consolidation_examples[old_sku]["distributed_to"][new_sku] = df[old_sku_count_col].sum()

            df.drop(columns=[old_sku_count_col], inplace=True)

    new_skus = [
        "B10-1000", "C6-1000", "C8-1000", "N6-1000", "P4-1000", "S10-1000",
        "HR12-1000", "SH10-1000", "SH8-1000", "SH6-1000", "SH4-1000",
        "SHN6-1000", "SH10E-1000", "SH6E-1000", "SH8E-1000", "SH4E-1000",
        "CK2-1000", "EVT-1000", "SS0-1000", "KB1-1000", "KS0-1000",
        "nan", "DH-1000", "DS-M", "DS-L", "FFS-1000", "EB5-1000"
    ]

    valid_count_columns = [f"{sku}_count" for sku in new_skus if f"{sku}_count" in df.columns]
    all_other_columns = [col for col in df.columns if not col.endswith('_count')]

    df_filtered = df[all_other_columns + valid_count_columns]

    return df_filtered

def last_minute_fixes(df):
    df['phone_area_code'] = df['phone_area_code'].replace('UNKNOWN', '')

    if 'coupon_1_id' in df.columns:
        df['used_coupon'] = df['coupon_1_id'].apply(lambda x: 1 if pd.notnull(x) and x != '' else 0)
    else:
        df['used_coupon'] = 0  # Assume 0 if the coupon_1_id column doesn't exist

    drop_cols = [
        "coupon_1_id", "coupon_2_id", "coupon_2_code", "coupon_2_amount", "coupon_2_description",
        "coupon_3_id", "coupon_3_code", "coupon_3_amount", "coupon_3_description",
        "coupon_4_id", "coupon_4_code", "coupon_4_amount", "coupon_4_description",
        "coupon_5_id", "coupon_5_code", "coupon_5_amount", "coupon_5_description"
    ]

    drop_cols2 = ['download_permissions_granted', 'shipping_1_id', 'shipping_1_total', 'shipping_2_total']

    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    df.drop(columns=[col for col in drop_cols2 if col in df.columns], inplace=True)

    df['repeat_customer'] = df.groupby('person_id')['person_id'] \
        .transform(lambda x: [1] * (len(x) - 1) + [0] if len(x) > 1 else [0])

    return df


def encode(df):
    df['has_billing_company'] = df['has_billing_company'].map({True: 1, False: 0})
    df['has_shipping_company'] = df['has_shipping_company'].map({True: 1, False: 0})
    df['is_income_censored'] = df['is_income_censored'].map({True: 1, False: 0})

    df['gender_binary'] = df['gender'].map({'M': 1, 'F': 0})
    df.loc[df['gender'] == 'UNKNOWN', 'gender_binary'] = np.nan
    df.loc[df['gender'] == 'Unknown', 'gender_binary'] = np.nan
    df.drop(columns='gender', inplace=True)


    one_hot_cols = [
        'order_currency', 'payment_method', 'shipping_method',
        'billing_state', 'billing_country',
        'shipping_state', 'shipping_country',
        'billing_zip_main', 'shipping_zip_main',
        'email_domain', 'email_category',
        'phone_area_code', 'most_frequent_product', 'billing_company_category'
    ]

    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)

    def cyclical_encode(df, col, max_val):
        df[f'{col}_sin'] = np.sin(2 * math.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * math.pi * df[col] / max_val)

    cyclical_encode(df, 'order_day', 31)
    cyclical_encode(df, 'order_month', 12)
    cyclical_encode(df, 'order_hour', 24)
    cyclical_encode(df, 'order_minute', 60)

    return df

def add_necessary_features(original_df, test_df):
    reference_column = "repeat_purchase_amount"
    if reference_column in original_df.columns:
        feature_start_index = original_df.columns.get_loc(reference_column) + 1
        original_features = original_df.columns[feature_start_index:]
    else:
        raise ValueError(f"Reference column '{reference_column}' not found in original CSV.")

    missing_features = [col for col in original_features if col not in test_df.columns]

    missing_data = pd.DataFrame(0, index=test_df.index, columns=missing_features)

    for feature in missing_features:
        if original_df[feature].dropna().isin([True, False]).all():
            missing_data[feature] = False  
        elif original_df[feature].dropna().isin([0, 1]).all():
            missing_data[feature] = 0 

    display_cols = ["billing_first_name", "billing_last_name", "billing_email"]
    preserved_display_data = test_df[display_cols] if all(col in test_df.columns for col in display_cols) else pd.DataFrame()

    test_df = pd.concat([test_df, missing_data], axis=1)

    final_test_df = test_df.reindex(columns=list(original_df.columns) + display_cols, fill_value=0)

    if not preserved_display_data.empty:
        final_test_df[display_cols] = preserved_display_data

    return final_test_df



def preprocess_data(df, original_df):
    df = df.join(df['date'].apply(parse_date))
    df.drop(columns=['date'], inplace=True)

    df['billing_phone'] = df['billing_phone'].apply(standardize_phone)


    df[['billing_zip_main', 'billing_zip_extra']] = df['billing_postcode'].apply(standardize_postcode)
    df[['shipping_zip_main', 'shipping_zip_extra']] = df['shipping_postcode'].apply(standardize_postcode)
    df.drop(columns=['billing_postcode', 'shipping_postcode'], inplace=True)


    df['billing_state'] = df['billing_state'].apply(standardize_state)
    df['shipping_state'] = df['shipping_state'].apply(standardize_state)


    df = parse_items(df)
    df = calculate_repeat_purchases(df)
    df = clean_income_data(df)
    df['phone_area_code'] = df['billing_phone'].apply(extract_area_code)
    df = categorize_email(df)
    df = categorize_billing_company(df)
    df = categorize_shipping_company(df)
    df = sort_and_identify_repeat_customers(df)
    df = transform_item_features(df)
    df = add_past_order_info(df)
    df = drop_cols(df)
    df = remove_item_subtotal(df)
    df = sku_condense(df)
    df = last_minute_fixes(df)
    df = encode(df)

    drop_cols_arr = ['order_datetime', 'repeat_purchase_amount', 'billing_city', 'shipping_city',
                     'order_day', 'order_month', 'order_hour', 'order_minute', 'billing_city_frequency',
                     'shipping_city_frequency']
    df.drop(columns=[col for col in drop_cols_arr if col in df.columns], inplace=True)

    df = add_necessary_features(original_df, df)

    return df


def clean_data(raw_df):
    census_data_path = "ACSST5Y2022/filtered_data.csv"
    original_training_data_path = "original_training_data.csv"

    original_df = pd.read_csv(original_training_data_path)

    merged_data = load_and_merge_data(raw_df, census_data_path)

    gender_lookup = fetch_gender_data()
    enriched_data = add_gender_info(merged_data, gender_lookup)

    return preprocess_data(enriched_data, original_df)

