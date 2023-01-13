import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute, Params, Router } from '@angular/router';
import { Subscription } from 'rxjs';
import * as productsData from '../../data/products.json';
import * as productsInfoData from '../../data/products-info.json';

@Component({
  selector: 'app-page-product',
  templateUrl: './page-product.component.html',
  styleUrls: ['./page-product.component.css'],
})
export class PageProductComponent implements OnInit, OnDestroy {
  private paramsSubscription: Subscription;
  private id: string;

  isLoading = true;
  product: any;
  productInfo: any;

  constructor(private route: ActivatedRoute, private router: Router) {}
  ngOnInit(): void {
    this.id = this.route.snapshot.params['product'];
    this.paramsSubscription = this.route.params.subscribe((params: Params) => {
      this.id = params['product'];
      this.product = (productsData as any).default.find(
        (item) => item.id === this.id
      );
      this.productInfo = (productsInfoData as any).default.find(
        (item) => item.id === this.id
      );

      if (!this.productInfo) {
        this.router.navigate(['not-found'], { replaceUrl: true });
      } else {
        this.isLoading = false;
      }
    });
  }
  ngOnDestroy() {
    this.paramsSubscription.unsubscribe();
  }
}
