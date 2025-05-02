from pymongo import ASCENDING, DESCENDING

class MongoDBHandler:
    def __init__(self, db):
        """
        Initialize with an existing pymongo database object.
        :param db: pymongo.database.Database instance
        """
        self.db = db

    def insert_one(self, collection, data):
        """
        Insert a single document into a collection.
        :param collection: Name of the collection
        :param data: Dictionary to insert
        """
        return self.db[collection].insert_one(data)

    def insert_many(self, collection, data_list):
        """
        Insert multiple documents into a collection.
        :param collection: Name of the collection
        :param data_list: List of dictionaries to insert
        """
        return self.db[collection].insert_many(data_list)

    def find_one(self, collection, query):
        """
        Find a single document that matches the query.
        :param collection: Name of the collection
        :param query: Query dictionary
        """
        return self.db[collection].find_one(query)

    def find_many(self, collection, query={}, sort_by=None, direction=ASCENDING, limit=0):
        """
        Find multiple documents that match the query.
        Supports sorting and limiting results.
        :param collection: Name of the collection
        :param query: Query dictionary
        :param sort_by: Field name to sort by (optional)
        :param direction: ASCENDING or DESCENDING
        :param limit: Max number of results to return (0 means no limit)
        """
        cursor = self.db[collection].find(query)
        if sort_by:
            cursor = cursor.sort(sort_by, direction)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def update_one(self, collection, query, update, upsert=False):
        """
        Update a single document that matches the query.
        :param collection: Name of the collection
        :param query: Query to match the document
        :param update: Update operation (e.g. {'$set': {...}})
        :param upsert: If True, insert document if it doesn't exist
        """
        return self.db[collection].update_one(query, update, upsert=upsert)

    def update_many(self, collection, query, update):
        """
        Update all documents that match the query.
        :param collection: Name of the collection
        :param query: Query dictionary
        :param update: Update operation
        """
        return self.db[collection].update_many(query, update)

    def delete_one(self, collection, query):
        """
        Delete a single document that matches the query.
        :param collection: Name of the collection
        :param query: Query dictionary
        """
        return self.db[collection].delete_one(query)

    def delete_many(self, collection, query):
        """
        Delete all documents that match the query.
        :param collection: Name of the collection
        :param query: Query dictionary
        """
        return self.db[collection].delete_many(query)

    def count_documents(self, collection, query={}):
        """
        Count the number of documents matching the query.
        :param collection: Name of the collection
        :param query: Query dictionary (default is empty = all documents)
        """
        return self.db[collection].count_documents(query)

    def drop_collection(self, collection):
        """
        Drop (delete) the entire collection.
        :param collection: Name of the collection
        """
        return self.db.drop_collection(collection)

    def list_collections(self):
        """
        List all collection names in the current database.
        """
        return self.db.list_collection_names()
