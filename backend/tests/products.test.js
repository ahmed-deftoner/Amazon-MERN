import request from "supertest";
import {app} from "../server";

describe("products", () => {
    describe("get products", () => {
        describe("given the product does not exist", () => {
            it("expect 404",async () => {
                const productId = "63382840e1cd226938c2d01f";
                await request(app).get(`/api/products/${productId}`).expect(404);
            });
        });

        describe("get all products", () => {
            it("should return a 200 status and the products", async () => {
      
              await request(app).get(`/api/products`).expect(200);
            });
        });

        describe("query products by seller", () => {
            it("should return a products by seller", async () => {
      
                const sellerID = "637aa4114558af068aabf638"
                await request(app)
                .get("/api/products/?seller=" + sellerID)
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].seller._id).toBe(sellerID)
                })
            });
        });

        describe("query products by category", () => {
            it("should return a product by category", async () => {
      
                const category = "Shirts"
                await request(app)
                .get("/api/products/?category=" + category)
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].category).toBe(category)
                })
            });
        });       
        
        describe("query products by name", () => {
            it("should return a products by name", async () => {
      
                const name = "Lacoste"
                const expected = "Lacoste Free Shirt"
                await request(app)
                .get("/api/products/?name=" + name)
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].name).toBe(expected)
                })
            });
        });

        describe("get top rated product", () => {
            it("should return a top rated product", async () => {
      
                const rating = 5
                await request(app)
                .get("/api/products/?order=toprated")
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].rating).toBe(rating)
                })
            });
        });

        describe("product with lowest price", () => {
            it("should return a products by lowest price", async () => {
      
                const price = 65
                await request(app)
                .get("/api/products/?order=lowest")
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].price).toBe(price)
                })
            });
        });

        describe("product with highest price", () => {
            it("should return a products by highest price", async () => {
      
                const price = 220
                await request(app)
                .get("/api/products/?order=highest")
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].price).toBe(price)
                })
            });
        });

        describe("check pagination of products route", () => {
            it("gives products on page provided", async () => {
                const page = 2
                await request(app)
                .get("/api/products/?pageNumber=" + page)
                .expect(200)
                .then((response) => {
                    expect(response.body.page).toBe(page)
                })
            });
        });

        describe("get products within price range", () => {
            it("gives products between a price range", async () => {
                const min = 68
                const max = 90
                const price = 78
                await request(app)
                .get("/api/products/?min=" + min + "&max=" + max)
                .expect(200)
                .then((response) => {
                    expect(response.body.products[0].price).toBe(price)
                })
            });
        });
    });
});