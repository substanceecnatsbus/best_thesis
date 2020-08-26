from crowick import create_app
import sys

sys.path.append("yolo/")

def main():
    app = create_app()
    print("yo")
    # 80 - http
    # 443 - https
    app.run(debug=False, host='0.0.0.0', port=int("80"))
    #  ssl_context=('ssl/crowick_me.crt', 'ssl/crowick_me.key')

if __name__ == '__main__':
    main()