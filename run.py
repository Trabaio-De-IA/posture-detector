from app import create_app


def main():
    app = create_app()
    print("[OK] Servidor iniciado em http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
