sh standalone_embed.sh start

docker build --file dockerfile.server -t name_variants_image .
docker stop name_variants && docker rm name_variants
docker run -d --name name_variants --net consultant_ai -p 8000:8000 name_variants_image