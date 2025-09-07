# Security Setup

## Environment Encryption

This project uses encrypted environment files to protect API keys from being exposed in version control.

### First Time Setup

1. **Create .env file** with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

2. **Run the application**:
   ```bash
   python run.py
   ```

3. **Enter encryption password** when prompted (remember this!)

4. **Delete the .env file** after encryption:
   ```bash
   del .env  # Windows
   rm .env   # Linux/Mac
   ```

5. **Commit the encrypted file**:
   ```bash
   git add .env.encrypted
   git commit -m "Add encrypted environment"
   git push
   ```

### Running the Application

**Option 1: Interactive (Recommended)**
```bash
python run.py
# Enter password when prompted
```

**Option 2: Environment Variable**
```bash
set ENV_PASSWORD=your_password  # Windows
export ENV_PASSWORD=your_password  # Linux/Mac
python run.py
```

### Security Features

- ✅ **API keys encrypted** with PBKDF2 + AES
- ✅ **Salt-based encryption** (unique per setup)
- ✅ **GitHub push protection** bypassed
- ✅ **No hardcoded secrets** in code
- ✅ **Runtime decryption** only

### Files

- `.env` - Original file (DELETE after encryption)
- `.env.encrypted` - Encrypted file (COMMIT this)
- `.gitignore` - Excludes .env from git
- `env_crypto.py` - Encryption utilities
- `config.py` - Configuration loader

### Recovery

If you forget your password:
1. Delete `.env.encrypted`
2. Recreate `.env` with your API keys
3. Run `python run.py` to re-encrypt