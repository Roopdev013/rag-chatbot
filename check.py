content = open('app.py').read()
if 'st.sidebar' in content:
    print("Sidebar code found!")
else:
    print("Sidebar code NOT found!")