#!/usr/bin/env python3
import pandas as pd
import psycopg2
from psycopg2 import sql
import os
from datetime import datetime
import re
import numpy as np # Import numpy for handling NaN

# Database connection parameters
# These should be adjusted to your specific PostgreSQL setup
DB_PARAMS = {
    'host': 'localhost',
    'database': 'invoice_db_2',
    'user': 'postgres',
    'password': 'Beta.dev$02', # Consider using environment variables or a secure method
    'port': 5432
}

def create_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def drop_tables(conn):
    """Drop existing tables in the correct order to avoid FK constraints"""
    try:
        cursor = conn.cursor()
        print("Dropping existing tables (if they exist)...")
        # Drop tables in reverse order of creation / dependency
        cursor.execute("DROP TABLE IF EXISTS purchase_order_line_items;")
        cursor.execute("DROP TABLE IF EXISTS purchase_orders;")
        cursor.execute("DROP TABLE IF EXISTS Items;") # Added Items
        cursor.execute("DROP TABLE IF EXISTS Vendors;")
        cursor.execute("DROP TABLE IF EXISTS Region;")
        cursor.execute("DROP TABLE IF EXISTS Customer;")
        cursor.execute("DROP TABLE IF EXISTS sales_transaction;")
        conn.commit()
        print("Existing tables dropped.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error dropping tables: {e}")


def create_tables(conn):
    """Create database tables for Vendors, Items, POs, and PO Line Items"""
    try:
        cursor = conn.cursor()

        print("Creating tables...")

        # Create Vendors table (matching vendor.csv)
        cursor.execute("""
        CREATE TABLE Vendors (
            VendorID INT PRIMARY KEY,
            VendorName VARCHAR(255) NOT NULL,
            ContactPerson VARCHAR(255),
            Email VARCHAR(255),
            PhoneNumber VARCHAR(50),
            AddressLine1 VARCHAR(255),
            AddressLine2 VARCHAR(255),
            City VARCHAR(100),
            State VARCHAR(100),
            PostalCode VARCHAR(20),
            Country VARCHAR(100),
            Region VARCHAR(100),
            TaxID VARCHAR(100),
            DefaultPaymentTerms VARCHAR(50),
            VendorRating DECIMAL(3, 1),
            CreatedAt TIMESTAMP,
            UpdatedAt TIMESTAMP
        );
        """)
        print("Created Vendors table.")

        # Create Items table (matching items.csv)
        cursor.execute("""
        CREATE TABLE Items (
            ItemID INT PRIMARY KEY,
            HSN VARCHAR(50) UNIQUE,
            ItemName VARCHAR(255) NOT NULL,
            ItemDescription TEXT,
            Category VARCHAR(100),
            UnitOfMeasure VARCHAR(50),
            StandardUnitPrice DECIMAL(12, 2),
            CreatedAt TIMESTAMP,
            UpdatedAt TIMESTAMP
        );
        """)
        print("Created Items table.")

        # Create purchase_orders table (matching purchase_order.csv)
        cursor.execute("""
        CREATE TABLE purchase_orders (
            PO_ID INT PRIMARY KEY,
            PONumber VARCHAR(50) UNIQUE,
            VendorID INT,
            OrderDate DATE,
            ExpectedDeliveryDate DATE,
            ShippingAddressLine1 VARCHAR(255),
            ShippingAddressLine2 VARCHAR(255),
            ShippingCity VARCHAR(100),
            ShippingState VARCHAR(100),
            ShippingPostalCode VARCHAR(20),
            ShippingCountry VARCHAR(100),
            PaymentTerms VARCHAR(50),
            SubtotalAmount DECIMAL(15, 2),
            TaxAmount DECIMAL(15, 2),
            ShippingCost DECIMAL(15, 2),
            TotalAmount DECIMAL(15, 2),
            Currency VARCHAR(3),
            Status VARCHAR(50),
            BuyerContactPerson VARCHAR(255),
            Notes TEXT,
            CreatedAt TIMESTAMP,
            UpdatedAt TIMESTAMP,
            FOREIGN KEY (VendorID) REFERENCES Vendors (VendorID)
        );
        """)
        print("Created purchase_orders table.")

        # Create purchase_order_line_items table (matching purchase_order_line_items.csv)
        cursor.execute("""
        CREATE TABLE purchase_order_line_items (
            POLineID INT PRIMARY KEY,
            PO_ID INT NOT NULL,
            LineNumber INT,
            ItemID INT,
            ItemDescription TEXT,
            QuantityOrdered INT,
            UnitOfMeasure VARCHAR(50),
            UnitPrice DECIMAL(15, 4),
            LineTotal DECIMAL(15, 2),
            LineExpectedDeliveryDate DATE NULL,
            QuantityReceived INT NULL,
            ActualDeliveryDate DATE NULL,
            LineStatus VARCHAR(50),
            CreatedAt TIMESTAMP,
            UpdatedAt TIMESTAMP,
            FOREIGN KEY (PO_ID) REFERENCES purchase_orders (PO_ID),
            FOREIGN KEY (ItemID) REFERENCES Items (ItemID)
        );
        """)
        print("Created purchase_order_line_items table.")

        cursor.execute("""
        CREATE TABLE Region (
            RegionID VARCHAR(10) PRIMARY KEY,
            RegionName VARCHAR(100) NOT NULL
        );
        """)
        print("Created Region table.")

        cursor.execute("""
        CREATE TABLE Customer (
            CustomerID VARCHAR(20) PRIMARY KEY,
            CustomerName VARCHAR(255) NOT NULL,
            CustomerType VARCHAR(50),
            City VARCHAR(100),
            Country VARCHAR(100)
        );
        """)
        print("Created Customer table.")

        cursor.execute("""
        CREATE TABLE sales_transaction (
            Transaction_ID INT PRIMARY KEY,
            Transaction_Date DATE,
            Product_ID INT,
            Vendor_ID INT,
            Purchase_Order_ID INT,
            Invoice_ID VARCHAR(20),
            Quantity_Sold INT,
            Sale_Price DECIMAL(15, 2),
            Total_Sale_Amount DECIMAL(15, 2),
            Customer_ID VARCHAR(20),
            Region_ID VARCHAR(10),
            Customer_Type VARCHAR(50),
            Year INT,
            Month INT,
            FOREIGN KEY (Product_ID) REFERENCES Items (ItemID),
            FOREIGN KEY (Vendor_ID) REFERENCES Vendors (VendorID),
            FOREIGN KEY (Purchase_Order_ID) REFERENCES purchase_orders (PO_ID),
            FOREIGN KEY (Customer_ID) REFERENCES Customer (CustomerID),
            FOREIGN KEY (Region_ID) REFERENCES Region (RegionID)
        );
        """)
        print("Created sales_transaction table.")

        conn.commit()
        print("All tables created successfully.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise # Re-raise the exception to stop execution if table creation fails

def safe_float(value):
    """Convert value to float, handling potential NaN or empty strings."""
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int(value):
    """Convert value to int, handling potential NaN, float, or empty strings."""
    if pd.isna(value) or value == '':
        return None
    try:
        # Handle cases where pandas might read integers as floats (e.g., if NaNs are present)
        return int(float(value))
    except (ValueError, TypeError):
        return None

def safe_date(date_str):
    """Convert date string to date object, handling potential NaT or empty strings."""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        # Handle different potential date formats if necessary, assuming YYYY-MM-DD for now
        return pd.to_datetime(date_str).date()
    except (ValueError, TypeError):
         # Try parsing timestamp format if date format fails
        try:
            return pd.to_datetime(date_str).date()
        except (ValueError, TypeError):
            print(f"Warning: Could not parse date: {date_str}")
            return None


def safe_timestamp(ts_str):
    """Convert timestamp string to datetime object, handling potential NaT or empty strings."""
    if pd.isna(ts_str) or ts_str == '':
        return None
    try:
        return pd.to_datetime(ts_str)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse timestamp: {ts_str}")
        return None

def load_vendors_data(conn, csv_file):
    """Load data into Vendors table from CSV file"""
    success_count = 0
    error_count = 0
    loaded_vendor_ids = set()
    try:
        cursor = conn.cursor()
        # Specify dtype to avoid issues with mixed types, especially for IDs/codes
        vendors_df = pd.read_csv(csv_file, dtype={'PostalCode': str, 'TaxID': str, 'PhoneNumber': str})
        # Replace numpy NaN with None for database insertion
        vendors_df = vendors_df.replace({np.nan: None})

        print(f"Loading {len(vendors_df)} rows from {csv_file} into Vendors...")

        for _, row in vendors_df.iterrows():
            vendor_id = safe_int(row.get('VendorID'))
            if vendor_id is None:
                print(f"Skipping row due to missing VendorID: {row.to_dict()}")
                error_count += 1
                continue

            # Handle potential missing values more robustly
            vendor_rating = safe_float(row.get('VendorRating'))
            created_at = safe_timestamp(row.get('CreatedAt'))
            updated_at = safe_timestamp(row.get('UpdatedAt'))


            insert_sql = sql.SQL("""
            INSERT INTO Vendors (VendorID, VendorName, ContactPerson, Email, PhoneNumber,
                               AddressLine1, AddressLine2, City, State, PostalCode, Country,
                               Region, TaxID, DefaultPaymentTerms, VendorRating, CreatedAt, UpdatedAt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (VendorID) DO NOTHING;
            """)

            values = (
                vendor_id,
                row.get('VendorName'),
                row.get('ContactPerson'),
                row.get('Email'),
                row.get('PhoneNumber'),
                row.get('AddressLine1'),
                row.get('AddressLine2'),
                row.get('City'),
                row.get('State'),
                row.get('PostalCode'),
                row.get('Country'),
                row.get('Region'),
                row.get('TaxID'),
                row.get('DefaultPaymentTerms'),
                vendor_rating,
                created_at,
                updated_at
            )

            try:
                cursor.execute(insert_sql, values)
                if cursor.rowcount > 0: # Check if a row was actually inserted
                    loaded_vendor_ids.add(vendor_id)
                    success_count += 1
                # else: Row already existed, not an error, but not counted as new success
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting VendorID {vendor_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback() # Rollback this specific transaction
                error_count += 1
                # Optionally continue to next row or break, depending on desired behavior
                continue

        conn.commit() # Commit all successful inserts
        print(f"Vendors data loaded: {success_count} successful, {error_count} errors.")
        return loaded_vendor_ids
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return set()
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading vendors data: {e}")
        return set() # Return empty set on critical error

def load_items_data(conn, csv_file):
    """Load data into Items table from CSV file"""
    success_count = 0
    error_count = 0
    loaded_item_ids = set()
    try:
        cursor = conn.cursor()
        items_df = pd.read_csv(csv_file)
        items_df = items_df.replace({np.nan: None}) # Replace numpy NaN with None

        print(f"Loading {len(items_df)} rows from {csv_file} into Items...")

        for _, row in items_df.iterrows():
            item_id = safe_int(row.get('ItemID'))
            if item_id is None:
                print(f"Skipping row due to missing ItemID: {row.to_dict()}")
                error_count += 1
                continue

            standard_unit_price = safe_float(row.get('StandardUnitPrice'))
            created_at = safe_timestamp(row.get('CreatedAt'))
            updated_at = safe_timestamp(row.get('UpdatedAt'))

            insert_sql = sql.SQL("""
            INSERT INTO Items (ItemID, HSN, ItemName, ItemDescription, Category,
                           UnitOfMeasure, StandardUnitPrice, CreatedAt, UpdatedAt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ItemID) DO NOTHING;
            """)

            values = (
                item_id,
                row.get('HSN'),
                row.get('ItemName'),
                row.get('ItemDescription'),
                row.get('Category'),
                row.get('UnitOfMeasure'),
                standard_unit_price,
                created_at,
                updated_at
            )

            try:
                cursor.execute(insert_sql, values)
                if cursor.rowcount > 0:
                    loaded_item_ids.add(item_id)
                    success_count += 1
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting ItemID {item_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Items data loaded: {success_count} successful, {error_count} errors.")
        return loaded_item_ids
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return set()
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading items data: {e}")
        return set()


def load_purchase_orders_data(conn, csv_file, valid_vendor_ids):
    """Load data into purchase_orders table from CSV file"""
    success_count = 0
    error_count = 0
    loaded_po_ids = set()
    try:
        cursor = conn.cursor()
        # Specify dtype for potential string IDs/codes
        po_df = pd.read_csv(csv_file, dtype={'ShippingPostalCode': str, 'PONumber': str})
        po_df = po_df.replace({np.nan: None})

        print(f"Loading {len(po_df)} rows from {csv_file} into purchase_orders...")

        for _, row in po_df.iterrows():
            po_id = safe_int(row.get('PO_ID'))
            vendor_id = safe_int(row.get('VendorID'))

            if po_id is None:
                print(f"Skipping row due to missing PO_ID: {row.to_dict()}")
                error_count += 1
                continue

            if vendor_id not in valid_vendor_ids:
                print(f"Skipping PO_ID {po_id} due to invalid or missing VendorID: {vendor_id}")
                error_count += 1
                continue

            # Safe data conversions
            order_date = safe_date(row.get('OrderDate'))
            expected_delivery_date = safe_date(row.get('ExpectedDeliveryDate'))
            subtotal = safe_float(row.get('SubtotalAmount'))
            tax = safe_float(row.get('TaxAmount'))
            shipping = safe_float(row.get('ShippingCost'))
            total = safe_float(row.get('TotalAmount'))
            created_at = safe_timestamp(row.get('CreatedAt'))
            updated_at = safe_timestamp(row.get('UpdatedAt'))

            insert_sql = sql.SQL("""
            INSERT INTO purchase_orders (PO_ID, PONumber, VendorID, OrderDate, ExpectedDeliveryDate,
                                       ShippingAddressLine1, ShippingAddressLine2, ShippingCity, ShippingState,
                                       ShippingPostalCode, ShippingCountry, PaymentTerms, SubtotalAmount,
                                       TaxAmount, ShippingCost, TotalAmount, Currency, Status,
                                       BuyerContactPerson, Notes, CreatedAt, UpdatedAt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (PO_ID) DO NOTHING;
            """)

            values = (
                po_id, row.get('PONumber'), vendor_id, order_date, expected_delivery_date,
                row.get('ShippingAddressLine1'), row.get('ShippingAddressLine2'), row.get('ShippingCity'), row.get('ShippingState'),
                row.get('ShippingPostalCode'), row.get('ShippingCountry'), row.get('PaymentTerms'), subtotal,
                tax, shipping, total, row.get('Currency'), row.get('Status'),
                row.get('BuyerContactPerson'), row.get('Notes'), created_at, updated_at
            )

            try:
                cursor.execute(insert_sql, values)
                if cursor.rowcount > 0:
                    loaded_po_ids.add(po_id)
                    success_count += 1
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting PO_ID {po_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Purchase orders data loaded: {success_count} successful, {error_count} errors.")
        return loaded_po_ids
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return set()
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading purchase orders data: {e}")
        return set()


def load_purchase_order_line_items_data(conn, csv_file, valid_po_ids, valid_item_ids):
    """Load data into purchase_order_line_items table from CSV file"""
    success_count = 0
    error_count = 0
    try:
        cursor = conn.cursor()
        poli_df = pd.read_csv(csv_file)
        poli_df = poli_df.replace({np.nan: None}) # Replace numpy NaN with None

        print(f"Loading {len(poli_df)} rows from {csv_file} into purchase_order_line_items...")

        for _, row in poli_df.iterrows():
            po_line_id = safe_int(row.get('POLineID'))
            po_id = safe_int(row.get('PO_ID'))
            item_id = safe_int(row.get('ItemID'))

            if po_line_id is None:
                print(f"Skipping row due to missing POLineID: {row.to_dict()}")
                error_count += 1
                continue

            if po_id not in valid_po_ids:
                print(f"Skipping POLineID {po_line_id} due to invalid or missing PO_ID: {po_id}")
                error_count += 1
                continue

            # ItemID might be optional in some systems, but required by our FK. Check if valid.
            if item_id not in valid_item_ids:
                 # Allow loading even if ItemID is missing/invalid in CSV, setting FK to NULL
                 # This assumes the ItemID column in the DB allows NULLs. If not, this row should be skipped.
                 # Adjusting table definition to allow NULL ItemID might be needed if source data is inconsistent.
                 # For now, we enforce the FK constraint and skip if ItemID is invalid.
                 print(f"Skipping POLineID {po_line_id} due to invalid or missing ItemID: {item_id}")
                 error_count += 1
                 continue


            # Safe data conversions
            line_number = safe_int(row.get('LineNumber'))
            quantity_ordered = safe_int(row.get('QuantityOrdered'))
            unit_price = safe_float(row.get('UnitPrice'))
            line_total = safe_float(row.get('LineTotal'))
            line_expected_delivery_date = safe_date(row.get('LineExpectedDeliveryDate'))
            quantity_received = safe_int(row.get('QuantityReceived'))
            actual_delivery_date = safe_date(row.get('ActualDeliveryDate'))
            created_at = safe_timestamp(row.get('CreatedAt'))
            updated_at = safe_timestamp(row.get('UpdatedAt'))


            insert_sql = sql.SQL("""
            INSERT INTO purchase_order_line_items (POLineID, PO_ID, LineNumber, ItemID, ItemDescription,
                                                 QuantityOrdered, UnitOfMeasure, UnitPrice, LineTotal,
                                                 LineExpectedDeliveryDate, QuantityReceived, ActualDeliveryDate,
                                                 LineStatus, CreatedAt, UpdatedAt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (POLineID) DO NOTHING;
            """)

            values = (
                po_line_id, po_id, line_number, item_id, row.get('ItemDescription'),
                quantity_ordered, row.get('UnitOfMeasure'), unit_price, line_total,
                line_expected_delivery_date, quantity_received, actual_delivery_date,
                row.get('LineStatus'), created_at, updated_at
            )

            try:
                cursor.execute(insert_sql, values)
                success_count += cursor.rowcount # Add 1 if row inserted, 0 otherwise
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting POLineID {po_line_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Purchase order line items data loaded: {success_count} successful, {error_count} errors.")
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading purchase order line items data: {e}")

def load_region_data(conn, csv_file):
    """Load data into Region table from CSV file"""
    success_count = 0
    error_count = 0
    loaded_region_ids = set()
    try:
        cursor = conn.cursor()
        region_df = pd.read_csv(csv_file)
        region_df = region_df.replace({np.nan: None})

        print(f"Loading {len(region_df)} rows from {csv_file} into Region...")

        for _, row in region_df.iterrows():
            region_id = row.get('Region_ID')
            if region_id is None:
                print(f"Skipping row due to missing Region_ID: {row.to_dict()}")
                error_count += 1
                continue

            insert_sql = sql.SQL("""
            INSERT INTO Region (RegionID, RegionName)
            VALUES (%s, %s)
            ON CONFLICT (RegionID) DO NOTHING;
            """)

            values = (
                region_id,
                row.get('RegionName')
            )

            try:
                cursor.execute(insert_sql, values)
                if cursor.rowcount > 0:
                    loaded_region_ids.add(region_id)
                    success_count += 1
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting Region_ID {region_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Region data loaded: {success_count} successful, {error_count} errors.")
        return loaded_region_ids
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return set()
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading region data: {e}")
        return set()

def load_customer_data(conn, csv_file):
    """Load data into Customer table from CSV file"""
    success_count = 0
    error_count = 0
    loaded_customer_ids = set()
    try:
        cursor = conn.cursor()
        customer_df = pd.read_csv(csv_file)
        customer_df = customer_df.replace({np.nan: None})

        print(f"Loading {len(customer_df)} rows from {csv_file} into Customer...")

        for _, row in customer_df.iterrows():
            customer_id = row.get('Customer_ID')
            if customer_id is None:
                print(f"Skipping row due to missing Customer_ID: {row.to_dict()}")
                error_count += 1
                continue

            insert_sql = sql.SQL("""
            INSERT INTO Customer (CustomerID, CustomerName, CustomerType, City, Country)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (CustomerID) DO NOTHING;
            """)

            values = (
                customer_id,
                row.get('CustomerName'),
                row.get('CustomerType'),
                row.get('City'),
                row.get('Country')
            )

            try:
                cursor.execute(insert_sql, values)
                if cursor.rowcount > 0:
                    loaded_customer_ids.add(customer_id)
                    success_count += 1
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting Customer_ID {customer_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Customer data loaded: {success_count} successful, {error_count} errors.")
        return loaded_customer_ids
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return set()
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading customer data: {e}")
        return set()

def load_sales_transaction_data(conn, csv_file, valid_item_ids, valid_vendor_ids, valid_po_ids, valid_customer_ids, valid_region_ids):
    """Load data into sales_transaction table from CSV file"""
    success_count = 0
    error_count = 0
    try:
        cursor = conn.cursor()
        sales_df = pd.read_csv(csv_file)
        sales_df = sales_df.replace({np.nan: None})

        print(f"Loading {len(sales_df)} rows from {csv_file} into sales_transaction...")

        for _, row in sales_df.iterrows():
            transaction_id = safe_int(row.get('Transaction_ID'))
            product_id = safe_int(row.get('Product_ID'))
            vendor_id = safe_int(row.get('Vendor_ID'))
            purchase_order_id = safe_int(row.get('Purchase_Order_ID'))
            customer_id = row.get('Customer_ID')
            region_id = row.get('Region_ID')

            if transaction_id is None:
                print(f"Skipping row due to missing Transaction_ID: {row.to_dict()}")
                error_count += 1
                continue

            # Validate foreign keys
            if product_id is not None and product_id not in valid_item_ids:
                print(f"Skipping Transaction_ID {transaction_id} due to invalid Product_ID: {product_id}")
                error_count += 1
                continue
            
            if vendor_id is not None and vendor_id not in valid_vendor_ids:
                print(f"Skipping Transaction_ID {transaction_id} due to invalid Vendor_ID: {vendor_id}")
                error_count += 1
                continue
                
            if purchase_order_id is not None and purchase_order_id not in valid_po_ids:
                print(f"Skipping Transaction_ID {transaction_id} due to invalid Purchase_Order_ID: {purchase_order_id}")
                error_count += 1
                continue
                
            if customer_id is not None and customer_id not in valid_customer_ids:
                print(f"Skipping Transaction_ID {transaction_id} due to invalid Customer_ID: {customer_id}")
                error_count += 1
                continue
                
            if region_id is not None and region_id not in valid_region_ids:
                print(f"Skipping Transaction_ID {transaction_id} due to invalid Region_ID: {region_id}")
                error_count += 1
                continue

            # Safe data conversions
            transaction_date = safe_date(row.get('Transaction_Date'))
            quantity_sold = safe_int(row.get('Quantity_Sold'))
            sale_price = safe_float(row.get('Sale_Price'))
            total_sale_amount = safe_float(row.get('Total_Sale_Amount'))
            year = safe_int(row.get('Year'))
            month = safe_int(row.get('Month'))

            insert_sql = sql.SQL("""
            INSERT INTO sales_transaction (
                Transaction_ID, Transaction_Date, Product_ID, Vendor_ID,
                Purchase_Order_ID, Invoice_ID, Quantity_Sold, Sale_Price,
                Total_Sale_Amount, Customer_ID, Region_ID, Customer_Type, Year, Month
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (Transaction_ID) DO NOTHING;
            """)

            values = (
                transaction_id, transaction_date, product_id, vendor_id,
                purchase_order_id, row.get('Invoice_ID'), quantity_sold, sale_price,
                total_sale_amount, customer_id, region_id, row.get('Customer_Type'), year, month
            )

            try:
                cursor.execute(insert_sql, values)
                success_count += cursor.rowcount
            except (Exception, psycopg2.Error) as e:
                print(f"Error inserting Transaction_ID {transaction_id}: {e}\nRow data: {row.to_dict()}")
                conn.rollback()
                error_count += 1
                continue

        conn.commit()
        print(f"Sales transaction data loaded: {success_count} successful, {error_count} errors.")
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(f"Critical error loading sales transaction data: {e}")

# Removed functions: is_valid_date, create_po_line_items_from_invoice_data,
# load_invoices_data, load_invoice_line_items_data, update_invoice_line_items_with_item_names


def main():
    # Create database connection
    conn = create_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return

    try:
        # Drop existing tables first to ensure a clean state
        drop_tables(conn)

        # Create the new table structures
        create_tables(conn)

        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Assume CompanyInvoiceData is in the same directory as the script
        data_dir = os.path.join(script_dir) # Simplified path logic
        if not os.path.exists(os.path.join(data_dir, "vendor.csv")):
             # If not found, try assuming script is one level up from CompanyInvoiceData
             potential_dir = os.path.join(os.path.dirname(script_dir), "CompanyInvoiceData")
             if os.path.exists(os.path.join(potential_dir, "vendor.csv")):
                 data_dir = potential_dir
             else:
                 # Try the original logic if the above fails
                if os.path.basename(script_dir) == "CompanyInvoiceData":
                    data_dir = script_dir
                else:
                    data_dir = os.path.join(script_dir, "CompanyInvoiceData")


        print(f"Looking for CSV files in: {data_dir}")

        # Define paths to the relevant CSV files
        vendors_csv = os.path.join(data_dir, "vendor.csv")
        items_csv = os.path.join(data_dir, "items.csv")
        purchase_orders_csv = os.path.join(data_dir, "purchase_order.csv")
        purchase_order_line_items_csv = os.path.join(data_dir, "purchase_order_line_items.csv")
        region_csv = os.path.join(data_dir, "region.csv")
        customer_csv = os.path.join(data_dir, "customer.csv")
        sales_transaction_csv = os.path.join(data_dir, "sales_transaction.csv")

        # Verify files exist
        files_to_check = [vendors_csv, items_csv, purchase_orders_csv, purchase_order_line_items_csv, 
                          region_csv, customer_csv, sales_transaction_csv]
        all_files_exist = True
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"Error: Required file does not exist: {file_path}")
                all_files_exist = False

        if not all_files_exist:
            print("One or more required CSV files are missing. Exiting.")
            return

        print("Loading data into tables...")
        # Load in correct order to satisfy foreign key constraints
        # Load parent tables first and get their valid IDs
        valid_vendor_ids = load_vendors_data(conn, vendors_csv)
        valid_item_ids = load_items_data(conn, items_csv)
        valid_region_ids = load_region_data(conn, region_csv)
        valid_customer_ids = load_customer_data(conn, customer_csv)

        # Load tables with foreign keys, passing the valid IDs for checking
        valid_po_ids = load_purchase_orders_data(conn, purchase_orders_csv, valid_vendor_ids)
        load_purchase_order_line_items_data(conn, purchase_order_line_items_csv, valid_po_ids, valid_item_ids)
        load_sales_transaction_data(conn, sales_transaction_csv, valid_item_ids, valid_vendor_ids, 
                                   valid_po_ids, valid_customer_ids, valid_region_ids)

        print("\nData loading process completed.")

    except Exception as e:
        print(f"An unexpected error occurred during the main process: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
