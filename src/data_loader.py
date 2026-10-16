import pandas as pd


def load_online_retail_ii(file_path):
    """
    Load both sheets of the Online Retail II dataset
    and combine them into one DataFrame.
    """

    excel_file = pd.ExcelFile(file_path)

    sheets = excel_file.sheet_names

    print("Available sheets:")
    for sheet in sheets:
        print("-", sheet)

    dataframes = []

    for sheet in sheets:
        print(f"\nLoading: {sheet}")

        df = pd.read_excel(
            file_path,
            sheet_name=sheet
        )

        print(
            f"{sheet}: "
            f"{df.shape[0]:,} rows, "
            f"{df.shape[1]} columns"
        )

        dataframes.append(df)

    retail = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"\nCombined dataset: "
        f"{retail.shape[0]:,} rows, "
        f"{retail.shape[1]} columns"
    )

    return retail