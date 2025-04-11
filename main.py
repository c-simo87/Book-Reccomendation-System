import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, simpledialog
from fuzzywuzzy import process
import seaborn as sn


# files
books_filename = 'BX-Books.csv'
users_filename = 'BX-Users.csv'
ratings_filename = 'BX-Book-Ratings.csv'
# Create the data frames
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# print(df_books.head())
# print(df_ratings.head())
# print(df_books.info())
# print(df_ratings.info())
# print(df_ratings.user.unique())

# RATINGS AND AGE DISTRIBUTION PLOTS
books = pd.read_csv('BX-Books.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
ratings.head()

pd.set_option('display.max_columns', None)
#print(books.head())

#DESCRIPTIVE METHODS
# Shows the ratings distribution. The vast majority are unevenly distribution, with the highest being 0
def ShowRatingsDistribution():
    plt.rc("font", size=15)
    ratings.bookRating.value_counts(sort=False).plot(kind='bar')
    plt.title('Rating Distribution\n')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('system1.png', bbox_inches='tight')
    plt.show()


# Shows the ratings provided by the age group. The most active ages are from 20-30
def ShowAgeDistribution():
    users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
    plt.title('Age Distribution\n')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('system2.png', bbox_inches='tight')
    plt.show()


# Preprocessing the data
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()

most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'],
                                index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary

average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()


def ShowAverageRating():
    plt.rc('font', size=15)
    # Add 'by' parameter to specify the column to use for sorting
    sn.regplot(x='ratingCount', y='bookRating', data=average_rating, scatter_kws={'s': 10})
    plt.title('Average Ratings by ISBN\n')
    plt.xlabel('ISBN')
    plt.ylabel('Average Rating')
    plt.savefig('system3.png', bbox_inches='tight')
    plt.show()


# YEAR OF PUBLICATION PLOT
# Convert 'yearOfPublication' column to numeric and handle invalid values
books['yearOfPublication'] = pd.to_numeric(books['yearOfPublication'], errors='coerce')

# Remove NaN values and values above the current year (assuming future years are invalid)
books = books.dropna(subset=['yearOfPublication'])
valid_years_mask = (books['yearOfPublication'] >= 1950) & (books['yearOfPublication'] <= 2024)
books = books[valid_years_mask]

# Aggregate the data into decades
books['decade'] = (books['yearOfPublication'] // 10) * 10

# Merge books dataframe with ratings dataframe
merged_data = pd.merge(books, ratings, on='ISBN', how='inner')

# Count the number of ratings per book
ratings_per_book = merged_data.groupby('ISBN').size()

# Filter out books with no ratings
books_with_ratings = books[books['ISBN'].isin(ratings_per_book.index)]

# Aggregate the data by decade for books with ratings
books_with_ratings_by_decade = books_with_ratings.groupby('decade').size()

# Plotting the distribution of books published over the decades along with the number of ratings
plt.figure(figsize=(12, 6))
books_with_ratings_by_decade.plot(kind='bar', color='skyblue', label='Books Published')
ratings_per_decade = merged_data.groupby('decade').size()
ratings_per_decade.plot(kind='line', color='orange', marker='o', label='Number of Ratings')


def ShowYearPublication():
    # Plotting the distribution of books published over the years
    plt.title('Distribution of Books Published and Number of Ratings Over the Decades')
    plt.xlabel('Decade')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('system3.png', bbox_inches='tight')
    plt.show()


# Call the function to display the plot
# ShowRatingsDistribution()
# ShowAgeDistribution()
# ShowYearPublication()


# Cleaning the data
# To ensure statistical significance, users with less than 200 ratings,
# and books with less than 100 ratings are excluded
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

# Ratings Matrix
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns

# print(ratings_pivot.shape)
# ratings_pivot.head()


combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()

combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

book_ratingCount = (combine_book_rating.
    groupby(by=['bookTitle'])['bookRating'].
    count().
    reset_index().
    rename(columns={'bookRating': 'totalRatingCount'})
[['bookTitle', 'totalRatingCount']]
    )
book_ratingCount.head()

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle',
                                                         how='left')
rating_with_totalRatingCount.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(book_ratingCount['totalRatingCount'].describe())

# print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
us_canada_user_rating.head()

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID',
                                                          values='bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

heatmap_data = us_canada_user_rating_matrix.toarray()
# Find rows and columns with all zero values
non_zero_rows = ~np.all(heatmap_data == 0, axis=1)
non_zero_columns = ~np.all(heatmap_data == 0, axis=0)

# HeatMap
sn.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'User Rating'}, vmin=0, vmax=10)
plt.title('User Rating Heatmap')
plt.xlabel('User ID')
plt.ylabel('Book Index')
# plt.show()

# NON-DESCRIPTIVE METHOD
# KNN Implementation
# Using the brute algorithm, we use the us and canada ratings due to memory consumption
# The GUI will allow the user to enter a book title and will search the list using the fuzzywuzzy
# built in string matching tool
# The function will search for the index and determine the nearest neighbor K from the pivot table
# and display it to the list to the GUI


model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)


def GetReccomendation(book_title_input):
    query = book_title_input.lower()
    choices = [title.lower() for title in us_canada_user_rating_pivot.index]
    book_title = book_title_input  # input('Enter book title: ')  # Input book title as a string

    matches = process.extractOne(query, choices)

    # Check if the match has sufficient similarity (adjust the threshold as needed)
    if matches[1] < 80:
        app.result_label.config(text='No matching title can be found')
        return []

    # Get the index of the entered book title
    query_index = choices.index(matches[0])  # us_canada_user_rating_pivot.index.get_loc(book_title)
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=6)
    r = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
            app.result_label.config(
                text='Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]],
                                                           distances.flatten()[i]))
            r.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])
    return r


# Creating the GUIS
class BookRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Recommendation App")

        self.label = ttk.Label(root, text="Enter book title:")
        self.label.pack(pady=20)

        self.entry = ttk.Entry(root)
        self.entry.pack(pady=20)

        self.button = ttk.Button(root, text='Get Recommendations', command=self.show_recommendations)
        self.button.pack(pady=20)

        self.result_label = ttk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.recommendations_listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=70)
        self.recommendations_listbox.pack(pady=20)

    def show_recommendations(self):
        book_title_input = self.entry.get()
        recommendations = GetReccomendation(book_title_input)

        self.recommendations_listbox.delete(0, tk.END)

        if recommendations is not None:
            for book in recommendations:
                self.recommendations_listbox.insert(tk.END, book)
        else:
            self.recommendations_listbox.insert(tk.END, "No close match found.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BookRecommendationApp(root)
    root.mainloop()
