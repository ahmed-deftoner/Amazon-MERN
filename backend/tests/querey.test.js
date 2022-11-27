import request from "supertest";
import {app} from "../server";

describe("products", () => {
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
});