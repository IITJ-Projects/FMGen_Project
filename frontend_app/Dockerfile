# frontend_app/Dockerfile
FROM nginx:stable-alpine
WORKDIR /usr/share/nginx/html
COPY index.html ./index.html
COPY main.js ./main.js
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]