import struct

def float2bin32(f):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))
def convert_label_enc_to_binary(data_to_steal):
    data_to_steal_binary = data_to_steal.applymap(lambda x: float2bin32(x).replace('b', ''))
    return data_to_steal_binary

def check_column(column):
    contains_negative_values = any(column < 0)
    #contains_non_decimal_strings = any(~column.apply(is_decimal_string))
    contains_fractional_values = any(column.apply(has_fraction))
    return contains_negative_values, contains_fractional_values #contains_non_decimal_strings

def convert_one_hot_enc_to_binary(data_to_steal, numerical_columns):
    #ONLY NUMERICAL COLUMNS WILL BE CONVERTED TO BINARY

    #FIRST, COLUMNS ARE SORTED (LEFT PART OF DF: CATEGORICAL COLS, RIGHT PART OF DF: NUMERICAL COLS)
    # Get the list of all column names in the DataFrame
    all_columns = data_to_steal.columns.tolist()
    # Identify categorical columns by excluding numerical columns
    categorical_columns = [col for col in all_columns if col not in numerical_columns]
    cat_cols = categorical_columns.copy()
    # Combine the lists to create a new column order
    new_column_order = categorical_columns + numerical_columns
    # Rearrange the DataFrame using the new column order
    data_to_steal_binary = data_to_steal[new_column_order]
    column_order = categorical_columns.copy()
    int_col_order = []
    float_column_order = []
    for col in numerical_columns:
        #CHECK WHICH COLUMNS CONTAIN FRACTIONAL VALUES OR NEGATIVE VALUES
        #THE VALUES IN THESE COLUMNS NEED TO BE CONVERTED TO A 32bit REPRESENTATION OF A FLOAT, ints CAN BE CONVERTED BY int()
        negative_values, fractional_values = check_column(data_to_steal_binary[col])
        if negative_values or fractional_values == True:
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: (float2bin32(x)))
            float_column_order.append(col)

        else:
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: int(x))
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: (bin(abs(x))[2:].replace('b', '')))
            column_order.append(col)
            int_col_order.append(col)

    # Sort the columns based on the condition
    column_order.extend(float_column_order)
    # Check if the specified column contains the letter 'b'
    data_to_steal_binary = data_to_steal_binary[column_order]
    num_cat_cols = len(all_columns) - len(numerical_columns)
    num_int_cols = len(int_col_order)
    num_float_cols = len(float_column_order)

    return data_to_steal_binary, cat_cols, int_col_order, float_column_order, num_cat_cols, num_int_cols, num_float_cols
