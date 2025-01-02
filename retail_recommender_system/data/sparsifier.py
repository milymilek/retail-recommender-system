import polars as pl
from matplotlib import pyplot as plt


class Sparsifier:
    def __init__(self, relations: pl.DataFrame, users: pl.DataFrame, items: pl.DataFrame, namings: dict):
        self.relations = relations
        self.users = users
        self.items = items

        self.user_id_col = namings["user_id"]
        self.item_id_col = namings["item_id"]
        self.user_id_map_col = namings["user_id_map"]
        self.item_id_map_col = namings["item_id_map"]
        self.date_col = namings["date"]

    def sparsify(self, frac: float):
        n_users = self.users.get_column(self.user_id_col).n_unique()
        n_items = self.items.get_column(self.item_id_col).n_unique()
        relations_n_users = self.relations.get_column(self.user_id_col).n_unique()
        relations_n_items = self.relations.get_column(self.item_id_col).n_unique()

        users_filtered = self.relations.select(self.user_id_map_col).unique().sample(fraction=frac)
        relations_filtered = self.relations.join(users_filtered, on=self.user_id_map_col, how="right")
        relations_filtered_n_users = relations_filtered.get_column(self.user_id_col).n_unique()
        relations_filtered_n_items = relations_filtered.get_column(self.item_id_col).n_unique()

        print(f"""(Users) Nunique: {n_users}
(Items) Nunique: {n_items}
(Relations, Users) Nunique: {relations_n_users} | diff: {n_users - relations_n_users}
(Relations, Items) Nunique: {relations_n_items} | diff: {n_items - relations_n_items}
(Relations, Users) Nunique (filtered): {relations_filtered_n_users} | diff: {n_users - relations_filtered_n_users}
(Relations, Items) Nunique (filtered): {relations_filtered_n_items} | diff: {n_items - relations_filtered_n_items}""")

        relations_filtered.shape
        _, ax = plt.subplots(1, 2, figsize=(10, 5))

        relations_cnt = self.relations.group_by(self.user_id_map_col).agg(pl.len()).sort("len", descending=True)
        relations_filtered_cnt = relations_filtered.group_by(self.user_id_map_col).agg(pl.len()).sort("len", descending=True)
        ax[0].hist(relations_cnt.select("len"), bins=500)
        ax[1].hist(relations_filtered_cnt.select("len"), bins=500)
        users_filtered = self.users.join(relations_filtered.select(self.user_id_col).unique(), on=self.user_id_col, how="inner").drop(
            self.user_id_map_col
        )
        items_filtered = self.items.join(relations_filtered.select(self.item_id_col).unique(), on=self.item_id_col, how="inner")
        users_filtered = users_filtered.with_columns(**{self.user_id_map_col: pl.arange(0, len(users_filtered), 1)})
        items_filtered = items_filtered.with_columns(**{self.item_id_map_col: pl.arange(0, len(items_filtered), 1)})

        users_id_map = users_filtered.select(self.user_id_col, self.user_id_map_col).unique()
        articles_id_map = items_filtered.select(self.item_id_col, self.item_id_map_col).unique()
        # for c, id_map in zip([self.user_id_col, "article_id"], [users_id_map, articles_id_map]):
        #     id_map.write_parquet(f".data/hm/intermediate/frac_{str(args.frac).replace('.', '_')}/{c}_map.parquet")

        relations_filtered = (
            relations_filtered.drop(self.user_id_map_col, self.item_id_map_col)
            .sort(self.date_col)
            .join(users_id_map, on=self.user_id_col, how="left")
            .join(articles_id_map, on=self.item_id_col, how="left")
        )

        return {"relations": relations_filtered, "users": users_filtered, "items": items_filtered}
