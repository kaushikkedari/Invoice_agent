# Fallback database schema for when vector database access fails

DATABASE_SCHEMA = """
vendor :Stores information about suppliers.
        vendorid (INT, PRIMARY KEY): Unique identifier for each vendor.
        vendorname (VARCHAR, NOT NULL): Name of the vendor.
        contactperson (VARCHAR): Primary contact person at the vendor.
        email (VARCHAR): Contact email address.
        phonenumber (VARCHAR): Contact phone number.
        addressline1, addressline2, city, state, postalcode, country (VARCHAR): Address details.
        region (VARCHAR): Geographic or business region.
        taxid (VARCHAR): Vendor's tax identification number.
        defaultpaymentterms (VARCHAR): Standard payment terms (e.g., "Net 30").
        vendorrating (DECIMAL): A numerical rating for the vendor.
        createdat, updatedat (TIMESTAMP): Timestamps for record creation and last update.

items : Stores information about products or services offered/purchased.
        itemid (INT, PRIMARY KEY): Unique identifier for each item.
        hsn (VARCHAR, UNIQUE): Harmonized System of Nomenclature or unique code for the item.
        itemname (VARCHAR, NOT NULL): Name of the item.
        itemdescription (TEXT): Detailed description of the item.
        category (VARCHAR): Category the item belongs to (e.g., "Office Supplies", "Hardware").
        unitofmeasure (VARCHAR): Unit used for quantity (e.g., "Each", "Box", "KG").
        standardunitprice (DECIMAL): Standard price per unit.
        createdat, updatedat (TIMESTAMP): Timestamps for record creation and last update.

region :Stores geographical or sales regions.
        regionid (VARCHAR, PRIMARY KEY): Unique identifier for the region (e.g., "R1", "R3").
        regionname (VARCHAR, NOT NULL): Name of the region (e.g., "India-West", "North America").

customer : Stores information about customers.
        customerid (VARCHAR, PRIMARY KEY): Unique identifier for the customer (e.g., "CUST001").
        customername (VARCHAR, NOT NULL): Name of the customer.
        customertype (VARCHAR): Type of customer (e.g., "B2B", "Retail").
        city, country (VARCHAR): Customer location details.

purchase_orders : Stores header information for purchase orders sent to vendors.
        po_id (INT, PRIMARY KEY): Unique identifier for the purchase order.
        ponumber (VARCHAR, UNIQUE): Human-readable purchase order number (e.g., "PO-2024-09001").
        vendorid (INT, FOREIGN KEY -> vendors): Links to the vendor the po was sent to.
        orderdate, expecteddeliverydate (DATE): Dates related to the order.
        shippingaddressline1, ..., shippingcountry (VARCHAR): Delivery address details.
        paymentterms (VARCHAR): Payment terms specific to this po.
        subtotalamount, taxamount, shippingcost, totalamount (DECIMAL): Financial details of the order.
        currency (VARCHAR): Currency used (e.g., "INR", "USD").
        status (VARCHAR): Current status of the po (e.g., "Sent", "Fully Received").
        buyercontactperson (VARCHAR): Person who placed the order.
        notes (TEXT): Additional notes about the po.
        createdAt, updatedat (TIMESTAMP): Timestamps for record creation and last update.

purchase_order_line_items : Stores details about individual items within each purchase order.
        polineid (INT, PRIMARY KEY): Unique identifier for each line item within a PO.
        po_id (INT, NOT NULL, FOREIGN KEY -> purchase_orders): Links to the parent purchase order.
        linenumber: (INT): Sequential number for the line item within the PO.
        itemid: (INT, FOREIGN KEY -> Items): Links to the specific item ordered.
        itemdescription (TEXT): Description of the item (potentially copied from Items or specific to the PO).
        quantityordered (INT): Quantity of the item ordered.
        unitofmeasure (VARCHAR): Unit for the quantity.
        unitprice (DECIMAL): Price per unit for this specific order line.
        linetotal (DECIMAL): Total cost for this line item (quantityordered * UnitPrice).
        lineexpecteddeliverydate, actualdeliveryDate (DATE): Delivery dates specific to the line item.
        quantityreceived (INT): Quantity actually received for this line item.
        linestatus (VARCHAR): Status of the individual line item (e.g., "Ordered", "Partially Received").
        createdat, updatedat (TIMESTAMP): Timestamps for record creation and last update.

sales_transaction : Stores details about individual sales transactions made to customers.
        transaction_id (INT, PRIMARY KEY): Unique identifier for the sales transaction.
        transaction_date (DATE): Date the sale occurred.
        product_id (INT, FOREIGN KEY -> Items): Links to the item that was sold.
        vendor_id (INT, FOREIGN KEY -> Vendors): Links to the original vendor of the product (useful for tracking source).
        purchase_order_id (INT, FOREIGN KEY -> purchase_orders): Links to the PO through which the sold item might have been acquired.
        invoice_id (VARCHAR): Identifier for the sales invoice generated for this transaction.
        quantity_sold (INT): Quantity of the item sold in this transaction.
        sale_price (DECIMAL): Price per unit at which the item was sold.
        total_sale_amount (DECIMAL): Total amount for this transaction line.
        customer_id (VARCHAR, FOREIGN KEY -> Customer): Links to the customer who bought the item.
        region_id (VARCHAR, FOREIGN KEY -> Region): Links to the sales region.
        customer_type (VARCHAR): Type of customer (potentially redundant if linked via Customer_ID, but present in the CSV).
        year, month (INT): Extracted year and month from the transaction date for easier reporting.

Relationships:
        - Each vendor can have multiple purchase orders but each purchase order can have only one vendor.(purchase_order.vendorid -> vendor.vendorid)
        - An item can appear on many different purchase order lines, but each purchase order line refers to one specific item (purchase_order_line_items.itemid -> items.itemid).
        - A purchase order contains multiple line items, but each line item belongs to exactly one purchase order (purchase_order_line_items.po_id -> purchase_orders.po_id).
        - customer <-> sales_transaction: One-to-Many. A customer can have multiple sales transactions, but each transaction is associated with one customer (sales_transaction.customer_id -> customer.customerid).
        - region <-> sales_transaction: One-to-Many. A region can encompass many sales transactions, but each transaction belongs to one region (sales_transaction.region_id -> region.regionid).
        - An item (product) can be sold in many different transactions, but each sales transaction line involves one specific product (sales_transaction.product_id -> items.itemid).
        - A vendor's products might be involved in many sales transactions (as the original supplier), and each sales transaction can link back to one vendor (sales_transaction.vendor_id
         -> vendors.vendorid). This links the sale back to the original supplier of the item.
        -  A purchase order used to acquire stock might fulfill multiple sales transactions over time. Each sales transaction can optionally link back to the purchase order used to acquire the item (sales_transaction.purchase_order_id -> purchase_orders.po_id).
        
          """ 