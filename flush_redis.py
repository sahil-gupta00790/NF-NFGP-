import redis
import os
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
client = redis.from_url(redis_url)
client.flushall()
print("Redis flushed successfully")