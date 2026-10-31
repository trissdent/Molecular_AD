from typing import Dict, Optional
from tqdm import tqdm
import asyncpg, asyncio, json


def json_dumps(data):
    if data is None: return data
    return json.dumps(data)


async def write_many_postgresql(query: str, args: list, connection_string: str) -> None:
    """
    Executes a parameterized SQL query with multiple sets of parameters to write data into a PostgreSQL database.

    Parameters:
    - query (str): The parameterized SQL query to execute.
    - args (list): List of tuples, where each tuple contains the parameters for a single execution of the query.
    - connection_string (str): PostgreSQL database connection string.

    Returns:
        None
    """

    conn = await asyncpg.connect(connection_string)
    await conn.executemany(query, args)
    await conn.close()


async def fetch_one_postgresql(
    query: str, args: tuple, connection_string: str
) -> Optional[Dict]:
    """
    Executes a parameterized SQL query with a single set of parameters to fetch a single row from a PostgreSQL database.

    Parameters:
    - query (str): The parameterized SQL query to execute.
    - args (tuple): Tuple containing the parameters for the query.
    - connection_string (str): PostgreSQL database connection string.

    Returns:
    - Optional[Dict]: A dictionary representing the fetched row, or None if no row is found.
    """

    conn = await asyncpg.connect(connection_string)
    value = await conn.fetchrow(query, *args)
    await conn.close()
    return dict(value) if value else None


async def fetch_many_postgresql(
    query: str, args: tuple, connection_string: str
) -> list:
    """
    Executes a parameterized SQL query with a single set of parameters to fetch multiple rows from a PostgreSQL database.

    Parameters:
    - query (str): The parameterized SQL query to execute.
    - args (tuple): Tuple containing the parameters for the query.
    - connection_string (str): PostgreSQL database connection string.

    Returns:
    - list: A list of dictionaries representing the fetched rows.
    """

    conn = await asyncpg.connect(connection_string)
    values = await conn.fetch(query, *args)
    await conn.close()
    return [dict(v) for v in values]



def insert_into_db(
        table_name, 
        data, data_fields, data_fields_need_json_dumps, data_fields_on_conflict, 
        conflict_handling = "update", connection_string = None):
    assert conflict_handling in ["update", "nothing", "update null only"]

    if conflict_handling == "update":
        conflict_handling_command = f"""ON CONFLICT ({",".join(data_fields_on_conflict)}) DO UPDATE 
    SET {",".join([f"{field}=COALESCE(EXCLUDED.{field}, {table_name}.{field})" for field in data_fields])}"""
    elif conflict_handling == "nothing":
        conflict_handling_command = f"""ON CONFLICT DO NOTHING"""
    elif conflict_handling == "update null only":
        conflict_handling_command = f"""ON CONFLICT ({",".join(data_fields_on_conflict)}) DO UPDATE 
    SET {",".join([f"{field}=COALESCE({table_name}.{field}, EXCLUDED.{field})" for field in data_fields])}"""
    else:
        conflict_handling_command = f"""ON CONFLICT DO NOTHING"""

    insert_query = f"""INSERT INTO {table_name}({",".join(data_fields)}) 
    VALUES ({",".join([f"${i+1}" for i in range(len(data_fields))])})
    {conflict_handling_command}"""

    asyncio.run(write_many_postgresql(
        query = insert_query,
        args = [[line.get(field) if field not in data_fields_need_json_dumps else json_dumps(line.get(field)) \
                    for field in data_fields] for line in data],
        connection_string = connection_string
    ))
    return True

def update_batch_to_database(data, table, data_fields, data_fields_on_conflict, data_fields_need_json_dumps, connection_string):
    print(f"num item is", len(data))

    batch_size = 5000

    pbar = tqdm(range(0, len(data), batch_size))

    for i in pbar:
        batch = data[i:i+batch_size]

        insert_into_db(
            table_name=table, 
            data = batch,
            data_fields = data_fields,
            data_fields_need_json_dumps=data_fields_need_json_dumps,
            data_fields_on_conflict=data_fields_on_conflict,
            conflict_handling="update",
            connection_string=connection_string
        )

    return True