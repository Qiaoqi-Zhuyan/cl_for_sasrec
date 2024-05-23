import pandas as pd

movies_file = r"ml-1m/movies.dat"
ratings_file = r"ml-1m/ratings.dat"
users_file = r"ml-1m/users.dat"

# uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
# users = pd.read_table(users_file, sep="::", header=None, names=uname, engine="python")
# print(users.head())

rname = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table(ratings_file, header=None, sep='::', names=rname, engine='python')
ratings_df = pd.DataFrame(ratings)
Users = dict()
for idx, row in ratings_df.iterrows():
    user_id = row["user_id"]
    if user_id not in Users:
        Users[user_id] = []
    Users[user_id].append([row["movie_id"], row["timestamp"]])

for user_id in Users.keys():
    Users[user_id].sort(key=lambda x: x[1])

f = open("rating_debug.txt", "w")
for user in Users.keys():
    for i in Users[user]:
        f.write("%d %d %d\n" % (user, i[0], i[1]))
f.close()



    # print(row["user_id"], row["movie_id"])
# df.to_csv("out.txt", sep="\t", index=None)





# mname = ['movie_id','title','genres']
# movies = pd.read_table(movies_file, header=None, sep='::', names=mname, encoding="ISO-8859-1" ,engine='python')
# print(movies.head())