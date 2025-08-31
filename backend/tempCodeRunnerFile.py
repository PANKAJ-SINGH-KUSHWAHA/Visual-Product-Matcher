            product_id = int(filename.split('.')[0])
            product_info = product_df[product_df['id'] == product_id].to_dict(orient='records')[0]
