import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path, chunksize=10000):
    """
    Load the dataset from a CSV file in chunks.
    """
    try:
        # Read the first chunk to initialize the DataFrame
        df_iter = pd.read_csv(file_path, chunksize=chunksize)
        df = next(df_iter)
        
        # Concatenate remaining chunks
        for chunk in df_iter:
            df = pd.concat([df, chunk], ignore_index=True)
        
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def downcast_dtypes(df):
    """
    Downcast numerical columns to reduce memory usage.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing duplicates.
    """
    # Display the number of missing values before cleaning
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    print("\n")

    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float32', 'float64', 'int32', 'int64']:
            # Fill numerical columns with the mean
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            # Fill categorical columns with the mode
            df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Display the number of missing values after cleaning
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    print("\n")

    return df

def generate_insights(df, sample_size=10000):
    """
    Generate insights and visualizations from the dataset.
    """
    if df is not None:
        print("Generating insights...")
        
        # Clean the data
        df = clean_data(df)

        # Downcast dtypes to save memory
        df = downcast_dtypes(df)

        # Sample the data if it's too large
        if len(df) > sample_size:
            df = df.sample(sample_size)
            print(f"Data sampled to {sample_size} rows for analysis.")

        # Display basic information about the dataset
        print("Dataset Information:")
        print(df.info())
        print("\n")

        # Display the first few rows of the dataset
        print("First few rows of the dataset:")
        print(df.head())
        print("\n")

        # Display summary statistics for numerical columns only
        print("Summary statistics (Numerical Columns):")
        numerical_df = df.select_dtypes(include=['float32', 'int32'])
        print(numerical_df.describe())
        print("\n")

        # Generate and display correlation matrix for numerical columns
        print("Correlation matrix (Numerical Columns):")
        corr_matrix = numerical_df.corr()
        print(corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        # Identify and visualize trends and patterns for numerical columns
        for column in numerical_df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=numerical_df, x=numerical_df.index, y=column)
            plt.title(f'Trend for {column}')
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.show()

        # Pairplot for numerical columns to identify relationships
        if len(numerical_df.columns) > 1:
            sns.pairplot(numerical_df)
            plt.suptitle('Pairplot of Numerical Columns')
            plt.show()

        # Analyze and visualize categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=column)
            plt.title(f'Count plot for {column}')
            plt.xticks(rotation=45)
            plt.show()

            # If there are numerical columns, create box plots to show distribution
            if len(numerical_df.columns) > 0:
                for num_col in numerical_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, x=column, y=num_col)
                    plt.title(f'Box plot of {num_col} by {column}')
                    plt.xticks(rotation=45)
                    plt.show()

    else:
        print("No dataset to generate insights from.")

if __name__ == "__main__":
    
    file_path = 'iris.csv'  
    df = load_dataset(file_path)
    generate_insights(df)
